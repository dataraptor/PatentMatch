#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright: Shamim Ahamed
"""
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from addict import Dict

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MeanPoolingLayer(nn.Module):
    def __init__(self):
        super(MeanPoolingLayer, self).__init__()
        self.pool = MeanPooling()
        self.fc = nn.Linear(768, 6)
        
    def forward(self, inputs, mask):
        last_hidden_states = inputs[0]
        feature = self.pool(last_hidden_states, mask)
        outputs = self.fc(feature)
        return outputs


def weight_init_normal(module, model):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
        

class FB3Model(nn.Module):
    def __init__(self, backbone):
        super(FB3Model, self).__init__()
        self.config = AutoConfig.from_pretrained(backbone, output_hidden_states=True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        self.model = AutoModel.from_pretrained(backbone, config=self.config)
        self.head = MeanPoolingLayer()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone);
        
    def _init_weights(self, layer):
        for module in layer.modules():
            init_fn = weight_init_normal
            init_fn(module, self)
            # print(type(module))
    
    def forward(self, inputs):
        outputs = self.model(**inputs)
        outputs = self.head(outputs, inputs['attention_mask'])
        return outputs

if __name__ == '__main__':
    model = FB3Model('microsoft/deberta-v3-base')
    