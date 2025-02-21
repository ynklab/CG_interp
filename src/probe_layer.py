import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class ProbedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.encoder_cached_activation = None
        
    def forward(self, src, src_mask = None, src_key_padding_mask = None, is_causal = False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        self.encoder_cached_activation = src
        return src

    @classmethod
    def from_layer(cls, layer):
        assert type(layer) == nn.modules.transformer.TransformerEncoderLayer
        res = cls(d_model = 512, nhead = 4, dim_feedforward = 2048, 
                  dropout = True, activation = 'relu')
        res.self_attn = layer.self_attn
        res.linear1 = layer.linear1
        res.linear2 = layer.linear2
        res.norm1 = layer.norm1
        res.norm2 = layer.norm2
        res.dropout1 = layer.dropout1
        res.dropout2 = layer.dropout2
        res.activation = layer.activation
        
        return res
    
    def activate_caching(self):
        self.encoder_cached_activation = None

    def get_cached_activation(self):
        return self.encoder_cached_activation
    
