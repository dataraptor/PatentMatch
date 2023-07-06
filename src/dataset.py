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


class USPPPMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, limit_data_samples):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df = pd.read_csv(data_path)
        if limit_data_samples != None: self.df = self.df[:limit_data_samples]
        

    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        score = row['score']
        x = row['anchor'] + '[SEP]' + row['target'] + '[SEP]'  + row['context']
        
        inputs = self.tokenizer(
            x, 
            add_special_tokens=True,
            max_length=self.max_length, padding='max_length',
            truncation=True,
            return_offsets_mapping=False
        )
        for k, v in inputs.items(): inputs[k] = torch.tensor(v, dtype=torch.long)
        label = torch.tensor(score, dtype=torch.float)
        return inputs, label



import pandas as pd
if __name__ == '__main__':
    from model import USPPPMModel
    model = USPPPMModel('microsoft/deberta-v3-base')
    
    ds = USPPPMDataset('./Data/train.csv', model.tokenizer, 133, 4)
    
    print(ds[0])
