#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright: Shamim Ahamed
"""
import pandas as pd
import os
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Dataset
from cleantext import clean
import torch
from addict import Dict


class NLPSimpleDataset(Dataset):
    def __init__(self, df, data_column, target_columns, transform, tokenizer, max_length, limit_data_samples):
        if limit_data_samples != None: df = df.head(limit_data_samples)
        
        self.texts = df[data_column].values
        self.labels = df[target_columns].values
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.__df = df
        
    @property
    def data(self): return self.__df
    
    @staticmethod
    def target_columns(self): raise NotImplementedError()
    
    @staticmethod
    def target_size(self): raise NotImplementedError()
    
    @staticmethod
    def read_data(): raise NotImplementedError()
    
    @staticmethod
    def collate_fn(): raise NotImplementedError()
    
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx], 
            max_length=self.max_length, padding='max_length',
            truncation=True,
            return_offsets_mapping=False
        )
        for k, v in inputs.items(): inputs[k] = torch.tensor(v, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return inputs, label
    

def fb3_cleantext(text):
    return clean(text,
        fix_unicode=True,              # fix various unicode errors
        no_emoji=True,
        to_ascii=True,                 # transliterate to closest ASCII representation
        lower=False,                   # lowercase text
        no_line_breaks=False,          # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=True,         # replace all phone numbers with a special token
        no_numbers=True,               # replace all numbers with a special token
        no_digits=True,                # replace all digits with a special token
        no_currency_symbols=True,      # replace all currency symbols with a special token
        no_punct=False,                # remove punctuations
        replace_with_punct="",         # instead of removing punctuations you may replace them
        replace_with_url="<url>",
        replace_with_email="<email>",
        replace_with_phone_number="<phone>",
        replace_with_number="<number>",
        replace_with_digit="0",
        replace_with_currency_symbol="<cur>",
        lang="en"                       # set to 'de' for German special handling
    ).strip()


class FB3Dataset(NLPSimpleDataset):
    __data_col = 'full_text'
    __target_cols = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
    
    def __init__(self, data, phase, cfg_ds, model, fold_no, random_state, limit_data_samples):
        assert phase in ['train', 'valid']
        if phase=='train': data=data[data['fold']!=fold_no].reset_index(drop=True)
        elif phase=='valid': data=data[data['fold']==fold_no].reset_index(drop=True)
        
        super(FB3Dataset, self).__init__(
            df = data,
            data_column = FB3Dataset.__data_col,
            target_columns = FB3Dataset.__target_cols,
            transform = None,
            tokenizer = model.tokenizer,
            max_length = cfg_ds.max_length,
            limit_data_samples = limit_data_samples,
        )
        
    @staticmethod
    def target_columns(): return FB3Dataset.__target_cols
    
    @staticmethod
    def target_size(): return len(FB3Dataset.__target_cols)
    
    @staticmethod
    def read_data(data_root, n_fold, cfg_ds, random_state):
        df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        Fold = MultilabelStratifiedKFold(
            n_splits = n_fold,
            shuffle = True, 
            random_state = random_state,
        )
        for n, (train_index, val_index) in enumerate(Fold.split(df,df[FB3Dataset.__target_cols])):
            df.loc[val_index, 'fold'] = int(n)
        df['fold'] = df['fold'].astype(int)
        return df
    
    @staticmethod
    def collate_fn(batch): None
    
    
if __name__ == '__main__':
    from model import FB3Model
    model = FB3Model('microsoft/deberta-v3-base')
    cfg_ds = {
        'nfold': 4,
        'seed': -1,
        'batch_size': 8,
        'drop_last_batch': True,
        'max_length': 1429, 
    }
    data = FB3Dataset.read_data('./Data', 4, Dict(cfg_ds), 42)
    ds = FB3Dataset(data, 'train', Dict(cfg_ds), model, 0, 42, 2)
    print(ds[0])
