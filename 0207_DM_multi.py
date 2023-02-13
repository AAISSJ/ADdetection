import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from collections import Counter
import pickle
import random
import argparse
import time
from datetime import datetime

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertConfig, BertModel,XLMTokenizer, XLMModel

class Arg:
    version = 1
    # data
    epochs: int = 1  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 350  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    a_hidden_size = 512 # BERT-base: 768, BERT-large: 1024, BERT paper setting
    t_hidden_size = 768
    t_x_hidden_size = 1280
    batch_size: int = 8
            
class BertPooler(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
    
class Model(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        # config:
        
        self.args = args
        self.config = config
        self.batch_size = self.args.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.label_cols = 'label'
        self.label_names = ['Control','ProbableAD']
        self.num_labels = 2
        self.t_embed_type = self.config['t_embed']
        self.a_embed_type = self.config['a_embed']
        self.a_hidden = self.args.a_hidden_size
        

        if self.t_embed_type == "mbert":
            self.t_hidden = self.args.t_hidden_size
            
            t_pretrained = 'bert-base-multilingual-uncased'
            self.t_tokenizer = BertTokenizer.from_pretrained(t_pretrained)
            self.t_model = BertModel.from_pretrained(t_pretrained)
            
            
        elif self.t_embed_type == "xlm":
            self.t_hidden = self.args.t_x_hidden_size
            
            t_pretrained = 'xlm-mlm-100-1280'
            self.t_tokenizer = XLMTokenizer.from_pretrained(t_pretrained)
            self.t_model = XLMModel.from_pretrained(t_pretrained)
            self.pooler = BertPooler(self.t_hidden)
            
        self.hidden = int(self.a_hidden + self.t_hidden)
        
        if self.a_embed_type == "en":
            a_pretrained =  "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            
        elif self.a_embed_type == "gr":
            a_pretrained =  "lighteternal/wav2vec2-large-xlsr-53-greek"

        elif self.a_embed_type == "multi":
            a_pretrained = "voidful/wav2vec2-xlsr-multilingual-56"
            
        elif self.a_embed_type == "wv":
            a_pretrained ='facebook/wav2vec2-base'
            
        self.a_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(a_pretrained)
        self.a_model = Wav2Vec2Model.from_pretrained(a_pretrained)
        
        
        self.clf1 = nn.Linear(self.hidden, int(self.hidden/2))
        self.clf2 = nn.Linear(int(self.hidden/2), self.num_labels)
        
            
            
    def forward(self, text,audio):
        
        if self.t_embed_type == "mbert":
            t_out = self.t_model(text)[1] 

            
        elif self.t_embed_type == "xlm":
            t_out = self.t_model(text)[0]
            t_out = self.pooler(t_out)
            
            
        a_out = self.a_model(audio)['extract_features']#[2] #last_hidden_state , feature extraction
        a_out = a_out[:, 0, :] 
        
        #print(a_out)
        #print(a_out['extract_features'].shape) # ([8, 437, 512])
        #print(a_out['last_hidden_state'].shape) # ([8, 437, 1024]) => pooling 필요
        
        
        output = torch.cat((t_out,a_out),axis=1)   
        #print(output.shape)
        
        logits = self.clf2(self.clf1(output))
    
        return logits
        

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess_dataframe(self):
        
        tg_sr = 16000
        t_col_name = "transcript" 
        a_col_name = "audio_sr_16000"         
        df = pd.read_json('../Data/230126_total_asr_data.json')

        df[t_col_name] = df[t_col_name].map(lambda x: self.t_tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
            ))
        
    
        # 원래 길이: 562992, batch 16: 90000, batch 8: 140000
        # max_length=16000, truncation=True 이건 일단 돌려보고 결정 => 뒤쪽, 앞에쪽 뭐보면 좋을 지 그런거 check하면 좋으니까! 
        df[a_col_name] = df[a_col_name].map(lambda x: self.a_tokenizer(
            x,
            sampling_rate = tg_sr,
            max_length=100000, 
            truncation=True
            )['input_values'][0])
        
        df_train = df[df['ex'] == 'train']
        df_val = df[df['ex'] == 'eval']
        df_test = df[df['ex'] == 'test']

        print(f'# of train:{len(df_train)}, val:{len(df_val)}, test:{len(df_test)}')

        self.train_data = TensorDataset(
            torch.tensor(df_train[t_col_name].tolist(), dtype=torch.long),
            torch.tensor(df_train[a_col_name].tolist(), dtype=torch.float),
            torch.tensor(df_train[self.label_cols].tolist(), dtype=torch.long),
        )
        
        self.val_data = TensorDataset(
             torch.tensor(df_val[t_col_name].tolist(), dtype=torch.long),
             torch.tensor(df_val[a_col_name].tolist(), dtype=torch.float),
            torch.tensor(df_val[self.label_cols].tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(df_test[t_col_name].tolist(), dtype=torch.long),
             torch.tensor(df_test[a_col_name].tolist(), dtype=torch.float),
            torch.tensor(df_test[self.label_cols].tolist(), dtype=torch.long),
             torch.tensor(df_test.index.tolist(), dtype=torch.long),
        )

    def train_dataloader(self):
        
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        token, audio, labels = batch  
        logits = self(token, audio) 
        loss = nn.CrossEntropyLoss()(logits, labels)   
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        token, audio, labels = batch  
        logits = self(token, audio) 
        loss = nn.CrossEntropyLoss()(logits, labels)     
        
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }
            
    def test_step(self, batch, batch_idx):
        token, audio, labels,id_ = batch 
        print('id', id_)
        logits = self(token, audio) 
        
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'y_true': y_true,
            'y_pred': y_pred,
        }
    
    def validation_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(outputs)
        loss = float(_loss)
        y_true = []
        y_pred = []

        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            
        y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
        y_true = np.asanyarray(y_true)
        
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        
        val_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        
        self.log("val_acc", val_acc)
        
        print("-------val_report-------")
        metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                             target_names = self.label_names, 
                                             output_dict=True)
        df_result = pd.DataFrame(metrics_dict).transpose()
        pprint(df_result)
        
#         df_result.to_csv(
#             f'./result/{datetime.now().__format__("%m%d_%H%M")}_DM_MM_{self.t_embed_type}_{self.a_embed_type}_val.csv')

#         pred_df = pd.DataFrame(pred_dict)
#         pred_df.to_csv(
#             f'./result/{datetime.now().__format__("%m%d_%H%M")}_DM_MM_{self.t_embed_type}_{self.a_embed_type}_val_pred.csv')

        return {'loss': _loss}

    def test_epoch_end(self, outputs):

        y_true = []
        y_pred = []

        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            
        y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
        y_true = np.asanyarray(y_true)
        
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        
        
        print("-------test_report-------")
        metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                             target_names = self.label_names, 
                                             output_dict=True)
        df_result = pd.DataFrame(metrics_dict).transpose()
        pprint(df_result)
        
#         df_result.to_csv(
#             f'./result/{datetime.now().__format__("%m%d_%H%M")}_DM_MM_{self.t_embed_type}_{self.a_embed_type}_test.csv')

#         pred_df = pd.DataFrame(pred_dict)
#         pred_df.to_csv(
#             f'./result/{datetime.now().__format__("%m%d_%H%M")}_DM_MM_{self.t_embed_type}_{self.a_embed_type}_test_pred.csv')

    
def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything( config['random_seed'])
        
    model = Model(args,config) 
    model.preprocess_dataframe()

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=10,
        verbose=True,
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(),"checkpoints"),
        monitor='val_acc',
        auto_insert_metric_name=True,
        verbose=True,
        mode='max', 
        save_top_k=1,
      )    

    print(":: Start Training ::")
    #     
    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback,checkpoint_callback],
        enable_checkpointing = True,
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        deterministic=True, # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        # For GPU Setup
        gpus=[config['gpu']] if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model,dataloaders=model.test_dataloader(),ckpt_path="best")
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--random_seed", type=int, default=2023) 
    parser.add_argument("--t_embed", type=str, default="mbert") 
    parser.add_argument("--a_embed", type=str, default="xls") 
    
    
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__)       


"""

python 0207_DM_multi.py --gpu 1 --t_embed mbert --a_embed en
python 0207_DM_multi.py --gpu 1 --t_embed xlm --a_embed en

# don
python 0207_DM_multi.py --gpu 0 --t_embed xlm --a_embed gr
python 0207_DM_multi.py --gpu 1 --t_embed mbert --a_embed gr

"""