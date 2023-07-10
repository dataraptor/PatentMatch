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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold


table = """
A: Human Necessities
B: Operations and Transport
C: Chemistry and Metallurgy
D: Textiles
E: Fixed Constructions
F: Mechanical Engineering
G: Physics
H: Electricity
Y: Emerging Cross-Sectional Technologies
"""
splits = [i for i in table.split('\n') if i != '']
table = {e.split(': ')[0]: e.split(': ')[1] for e in splits}


class USPPPMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, limit_data_samples):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df = pd.read_csv(data_path)
        if limit_data_samples != None: self.df = self.df[:limit_data_samples]
        
    @staticmethod
    def read_data(data_root, n_fold, cfg_ds, random_state):
        df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        return df


    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        score = x['label']
        
        sep = '' + self.tokenizer.sep_token + ''
        # sig = x['context'][0]
        # cpc = '[' + sig + ']'
        
        # ctg = '[CTG] ' + table[sig]
        # ctx = '[CTX] ' + x['title']
        # anc = '[ANC] ' + x['anchor']
        # tgt = '[TGT] ' + x['target']
        # arr = [anc, tgt, ctx, ctg]
        
        
        #s = ' '.join(arr)
        s = x['anchor'] + sep + x['target'] + sep + x['title']
        #print(s)
        
        
        inputs = self.tokenizer(
            s, add_special_tokens=True,
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
    
    ds = USPPPMDataset('./Data/USPPPM 4Fold/fold-0-test.csv', model.tokenizer, 133, 4)
    
    print(ds[0])
