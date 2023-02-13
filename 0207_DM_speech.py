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

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class Arg:
    version = 1
    # data
    epochs: int = 50  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 500  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    hidden_size = 768 # BERT-base: 768, BERT-large: 1024, BERT paper setting
    batch_size: int = 8
            
    # 300/16 > 200/32 > 512/8
class Model(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        # config:
        
        self.args = args
        self.config = config
        self.batch_size = self.args.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.label_cols = 'label'# 'current_bp_emo_y' 
        self.label_names = ['Control','ProbableAD'] # ['bp_remission','bp_manic','bp_anxiety','bp_irritability','bp_depressed']
        self.num_labels = 2
        self.embed_type = self.config['embed']

        if self.embed_type == "en":
            pretrained =  "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            
        elif self.embed_type == "300":
            pretrained =  "facebook/wav2vec2-xls-r-300m"
            
        elif self.embed_type == "multi":
            pretrained =  "voidful/wav2vec2-xlsr-multilingual-56"
            
        elif self.embed_type == "gr":
            pretrained =  "lighteternal/wav2vec2-large-xlsr-53-greek"

        elif self.embed_type == "wv":
            pretrained ='facebook/wav2vec2-base'
            
        self.tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(pretrained)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(pretrained)
            
            
    def forward(self, input_ids, **kwargs):
    
        return self.model(input_ids).logits
        

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess_dataframe(self):
        
        tg_sr = 16000
        
        col_name = "audio_sr_16000"    
        df = pd.read_json('../Data/230126_total_asr_data.json')

        # 원래 길이: 562992, batch 16: 90000, batch 8: 140000
        # max_length=16000, truncation=True 이건 일단 돌려보고 결정 => 뒤쪽, 앞에쪽 뭐보면 좋을 지 그런거 check하면 좋으니까! 
        
        df[col_name] = df[col_name].map(lambda x: self.tokenizer(
            x,
            sampling_rate = tg_sr,
            max_length=140000, 
            truncation=True
            )['input_values'][0])
        
        df_train = df[df['ex'] == 'train']
        df_val = df[df['ex'] == 'eval']
        df_test = df[df['ex'] == 'test']

        print(f'# of train:{len(df_train)}, val:{len(df_val)}, test:{len(df_test)}')

        self.train_data = TensorDataset(
            torch.tensor(df_train[col_name].tolist(), dtype=torch.float),
            torch.tensor(df_train[self.label_cols].tolist(), dtype=torch.long),
        )
        
        self.val_data = TensorDataset(
             torch.tensor(df_val[col_name].tolist(), dtype=torch.float),
            torch.tensor(df_val[self.label_cols].tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(df_test[col_name].tolist(), dtype=torch.float),
            torch.tensor(df_test[self.label_cols].tolist(), dtype=torch.long),
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
        token, labels = batch  
        logits = self(input_ids=token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        token, labels = batch  
        logits = self(input_ids=token) 
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
        token, labels = batch
        logits = self(input_ids=token) 
        
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
#             f'./result/{datetime.now().__format__("%m%d_%H%M")}_DM_Speech_{self.embed_type}_val.csv')

#         pred_df = pd.DataFrame(pred_dict)
#         pred_df.to_csv(
#             f'./result/{datetime.now().__format__("%m%d_%H%M")}_{self.embed_type}_val_pred.csv')

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
        
        df_result.to_csv(
            f'./result/{datetime.now().__format__("%m%d_%H%M")}_DM_Speech_{self.embed_type}_test.csv')

        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(
            f'./result/{datetime.now().__format__("%m%d_%H%M")}_DM_Speech_{self.embed_type}_test_pred.csv')

    
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
    parser.add_argument("--embed", type=str, default="bert") 
    
    
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__)       


"""


# done
python 0207_DM_speech.py --gpu 0 --embed gr
python 0207_DM_speech.py --gpu 1 --embed en
python 0207_DM_speech.py --gpu 0 --embed 300
python 0207_DM_speech.py --gpu 0 --embed multi
python 0207_DM_speech.py --gpu 0 --embed wv

"""