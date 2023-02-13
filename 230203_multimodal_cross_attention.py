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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


# lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split

# text
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import XLMTokenizer, XLMModel

# Audio 
import librosa
import torchaudio
import transformers 
from transformers import Wav2Vec2FeatureExtractor, AutoModel



class Arg:
    version = 1
    # data
    epochs: int = 15  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 500  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    hidden_size = 256 # BERT-base: 768, BERT-large: 1024, BERT paper setting
    batch_size: int = 16
    

    # 300/16 > 200/32 > 512/8
class TextModel(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        # config:
        
        self.args = args
        self.config = config
        self.batch_size = self.args.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.embed_type = self.config['text_embed']
        self.hidden_size = self.args.hidden_size
        
        # dataset
        self.data = self.config['data_path']
        self.label_col = self.config['label_col'] # 'current_bp_emo_y' 
        self.label_names = ['Control','ProbableAD'] # ['bp_remission','bp_manic','bp_anxiety','bp_irritability','bp_depressed']
        self.num_labels = self.config['num_labels']
        self.col_name= self.config['text_col']
        
        # model 
        if self.embed_type == "bert":
            pretrained = "bert-base-uncased"
            self.tokenizer = BertTokenizer.from_pretrained(pretrained)
            self.model = BertModel.from_pretrained(pretrained)
            
        elif self.embed_type == "mbert":
            pretrained = 'bert-base-multilingual-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(pretrained)
            self.model = BertModel.from_pretrained(pretrained)
            
        elif self.embed_type == "xlm":
            pretrained = 'xlm-mlm-100-1280'
            self.tokenizer = XLMTokenizer.from_pretrained(pretrained)
            self.model = XLMModel.from_pretrained(pretrained)
        
        
        
    def forward(self, input_ids, **kwargs):
        '''
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        '''

        output= self.model(input_ids)
        last_hidden_states=output.last_hidden_state
        
        return last_hidden_states


class AudioModel(LightningModule):
    def __init__(self, args, config):
        super(AudioModel, self).__init__()

        self.args = args
        self.config = config
        self.batch_size = self.args.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.embed_type = self.config['audio_embed']
        self.hidden_size = self.args.hidden_size
         
        # dataset
        self.data = self.config['data_path']
        self.label_col = self.config['label_col'] # 'current_bp_emo_y' 
        self.label_names = ['Control','ProbableAD'] # ['bp_remission','bp_manic','bp_anxiety','bp_irritability','bp_depressed']
        self.num_labels = self.config['num_labels']
        self.col_name= self.config['audio_col']
        
        # model 
        if self.embed_type=='xlsr':
            pretrained="voidful/wav2vec2-xlsr-multilingual-56"
            self.tokenizer= Wav2Vec2FeatureExtractor.from_pretrained(pretrained)
            self.target_sampling_rate = self.tokenizer.sampling_rate
            self.model = AutoModel.from_pretrained(pretrained)
        
        elif self.embed_type=='wav2vec2':
            pretrained= "facebook/wav2vec2-base"
            self.tokenizer= Wav2Vec2FeatureExtractor.from_pretrained(pretrained)
            self.target_sampling_rate = self.tokenizer.sampling_rate
            self.model = AutoModel.from_pretrained(pretrained)



    def forward(self,input_ids, **kwargs):
        '''
        inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        '''
        
        with torch.no_grad():
            output=self.model(input_ids)
        last_hidden_states=output.last_hidden_state

        return last_hidden_states


class FusionModel(LightningModule):
    def __init__(self, args, config):
        super(FusionModel, self).__init__()
        
        self.config=config
        self.args=args
        
        # Data
        self.data=self.config['data_path']
        self.text_col = self.config['text_col']
        self.audio_col = self.config['audio_col']
        self.label_col = self.config['label_col']
        self.num_labels = self.config['num_labels']
        self.label_names = ['Control','ProbableAD']
        
        self.ta_nh = self.config['ta_nh']
        self.at_nh = self.config['at_nh']
        
        self.ta_dp = self.config['ta_dp']
        self.at_dp = self.config['at_dp']
        
        
        # meta data
        self.batch_size=self.args.batch_size
        self.hidden_size = self.args.hidden_size
    
        # Models 
        self.text_model = TextModel(args,config)
        self.audio_model = AudioModel(args,config)
        
        self.pool=nn.AdaptiveMaxPool2d((1, self.hidden_size)) #ex) [16, 500, 768] -> [16, 1, self.hidden_size] 
        
        self.mha_a_t = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads= self.ta_nh,dropout=self.at_dp)
        self.mha_t_a = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads= self.ta_nh ,dropout= self.ta_dp)
        
        self.dense1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, self.num_labels)
        
        

    def forward(self,text_input_ids, audio_input_ids, **kwargs):
        
        #print(":: Text Embeddings ::")
        text_embed=self.text_model(text_input_ids) # [16, seq_length, 768] 
        text_embed=self.pool(text_embed) # .reshape(text_embed.shape[0], -1) #  [16, self.hidden_size] 
        #print("text: ",text_embed.shape) 
        
        #print(":: Audio Embeddings ::")
        audio_embed=self.audio_model(audio_input_ids) # [16, seq_length, 1024] 
        audio_embed=self.pool(audio_embed) # .reshape(audio_embed.shape[0], -1) #  [16, self.hidden_size] 
        #print("audio: ",audio_embed.shape)

        # audio to text 
        x_a2t, _ = self.mha_a_t(text_embed, audio_embed, audio_embed) 
        x_a2t = torch.mean(x_a2t, dim=1)
        
        # text to audio  
        x_t2a, _ = self.mha_t_a(audio_embed, text_embed, text_embed) 
        x_t2a = torch.mean(x_t2a, dim=1)
        
    
        x_ta2 = torch.stack((x_a2t, x_t2a), dim=1) 
        x_ta2_mean, x_ta2_std = torch.std_mean(x_ta2, dim=1)
        x_ta2 = torch.cat((x_ta2_mean, x_ta2_std), dim=1) 
        fuse = x_ta2
        
        logits=self.dense3(self.dense2(self.dense1(fuse)))   
        
        return logits
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }


    def preprocess_dataframe(self):

        
        df = pd.read_json(self.data)

        df[self.text_col] = df[self.text_col].map(lambda x: self.text_model.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        
        # 원래 길이: 562992, batch 16: 90000, batch 8: 140000
        # max_length=16000, truncation=True 이건 일단 돌려보고 결정 => 뒤쪽, 앞에쪽 뭐보면 좋을 지 그런거 check하면 좋으니까! 
        
        df[self.audio_col] = df[self.audio_col].map(lambda x: self.audio_model.tokenizer(
            x,
            sampling_rate = self.audio_model.target_sampling_rate,
            max_length=140000, 
            truncation=True
        )['input_values'][0])
        
        
        df_train = df[df['ex'] == 'train']
        df_val = df[df['ex'] == 'eval']
        df_test = df[df['ex'] == 'test']

        print(f'# of train:{len(df_train)}, val:{len(df_val)}, test:{len(df_test)}')


        self.train_data = TensorDataset(
            torch.tensor(df_train[self.text_col].tolist(), dtype=torch.long),
            torch.tensor(df_train[self.audio_col].tolist(), dtype=torch.float),
            torch.tensor(df_train[self.label_col].tolist(), dtype=torch.long),
        )
        
        self.val_data = TensorDataset(
            torch.tensor(df_val[self.text_col].tolist(), dtype=torch.long),
            torch.tensor(df_val[self.audio_col].tolist(), dtype=torch.float),
            torch.tensor(df_val[self.label_col].tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(df_test[self.text_col].tolist(), dtype=torch.long),
             torch.tensor(df_test[self.audio_col].tolist(), dtype=torch.float),
             torch.tensor(df_test[self.label_col].tolist(), dtype=torch.long)
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
        text_token, audio_token, labels = batch  
        logits = self(text_input_ids=text_token, audio_input_ids=audio_token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        text_token, audio_token, labels = batch    
        logits = self(text_input_ids=text_token, audio_input_ids=audio_token)  
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
        text_token, audio_token, labels = batch
        logits = self(text_input_ids=text_token, audio_input_ids=audio_token) 
        preds = logits.argmax(dim=-1)

        y_pred = list(preds.cpu().numpy())
        labels= list(labels.cpu().numpy())
        
        return {
            'y_true': labels,
            'y_pred': y_pred
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
        # df_result.to_csv(
        #     f'./result/cross_attention_{datetime.now().__format__("%m%d_%H%M")}_{self.args.epochs}_{self.args.hidden_size}_{self.args.batch_size}_val.csv')


        # pred_df = pd.DataFrame(pred_dict)
        # pred_df.to_csv(
        #     f'./result/cross_attention_{datetime.now().__format__("%m%d_%H%M")}_{self.args.epochs}_{self.args.hidden_size}_{self.args.batch_size}_val_pred.csv')

        return {'loss': _loss}


    def test_epoch_end(self, outputs):
        y_pred = []
        y_true =[]

        for i in outputs:
            y_pred += i['y_pred']
            y_true += i['y_true']
        
        y_pred = np.asanyarray(y_pred)
        y_true = np.asanyarray(y_true)
        
        
        pred_dict = {}
        pred_dict['y_pred']= y_pred

        print(classification_report(y_true, y_pred))
        
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(
            f'./result/cross_attention_{datetime.now().__format__("%m%d_%H%M%S")}_{self.args.epochs}_{self.args.hidden_size}_{self.args.batch_size}_test_pred.csv')

    
def main(args,config):
    
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything( config['random_seed'])
    
    
    args.epochs = config['epochs']
    args.hidden_size = config['hidden_size']
    args.batch_size = config['batch_size']
    args.max_length = config['max_length']
    
    
    fusion_model=FusionModel(args, config)
    
    print(":: Processing Data ::")
    fusion_model.preprocess_dataframe()

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
     
    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback,checkpoint_callback],
        enable_checkpointing = True,
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        deterministic=False, # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        # For GPU Setup
        gpus=[config['gpu']] if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32
    )
    
    trainer.fit(fusion_model)

    trainer.test(fusion_model,dataloaders=fusion_model.val_dataloader(),ckpt_path="best")
    trainer.test(fusion_model,dataloaders=fusion_model.test_dataloader(),ckpt_path="best")
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # settings 
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=2023) 
    
    # models 
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="epoch")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=500)
    
    parser.add_argument("--ta_nh", type=int, default=2)
    parser.add_argument("--at_nh", type=int, default=2)
    parser.add_argument("--ta_dp", type=float, default=0.1)
    parser.add_argument("--at_dp", type=float, default=0.1)
    
    parser.add_argument("--text_embed", type=str, default="mbert") 
    parser.add_argument("--audio_embed", type=str, default="xlsr") 
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    
    
    
    # datasets 
    parser.add_argument("--data_path", type=str, default="../Data/230126_total_asr_data.json")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--text_col", type=str, default="transcript")
    parser.add_argument("--audio_col", type=str, default="audio_sr_16000")
    parser.add_argument("--num_labels", type=int, default=2)
    
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__)       

