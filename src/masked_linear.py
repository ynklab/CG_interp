# https://github.com/stevenxcao/subnetwork-probing/blob/main/code/masked_linear.py
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class L0Mask(nn.Module):
    def __init__(self, mask_dim, mask_p = 0.9):
        super().__init__()
        self.mask_setting = 'mask'
        self.mask_scores = nn.Parameter(torch.zeros(mask_dim))
        self.mask_p = mask_p
        self.l, self.r, self.b = -0.1, 1.1, 2 / 3
        self.threshold = 0.5
        self.init_weights()

    def init_weights(self):
        p = (self.mask_p - self.l) / (self.r - self.l)
        init.constant_(self.mask_scores, val=np.log(p / (1 - p)))
        
    def set_temperature(self, temp):
        self.b = temp
        
    def produce_mask(self, is_ablate):
        if self.training:
            u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / self.b)
        else:
            s = torch.sigmoid(self.mask_scores)
        s_bar = s * (self.r - self.l) + self.l
        mask = s_bar.clamp(min=0.0, max=1.0)
        if not self.training:
            if is_ablate:
                # reverse the mask
                mask = (mask < self.threshold).float()
            else:
                mask = (mask > self.threshold).float()
            
        return mask
    
    def regularizer(self):
        return torch.sum(torch.sigmoid(self.mask_scores - self.b * np.log(-self.l / self.r))) / self.mask_scores.numel()

class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 mask_p: float = 0.9, out_w_per_mask = 1, in_w_per_mask = 1, num_heads = 12, is_ablate=False):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.num_heads = num_heads
        self.out_w_per_mask = out_w_per_mask
        self.in_w_per_mask = in_w_per_mask
        
        assert out_features % out_w_per_mask == 0, "{} % {} not 0".format(out_features, out_w_per_mask)
        assert in_features % in_w_per_mask == 0, "{} % {} not 0".format(in_features, in_w_per_mask)
        mask_dim = (1, out_features // out_w_per_mask, 1, in_features // in_w_per_mask)
        self.mask = L0Mask(mask_dim)
        
        self.cached_activation = None
        self.do_caching = True
        self.is_ablate = is_ablate

    def produce_mask_reshaped(self):
        mask = self.mask.produce_mask(self.is_ablate)
        mask = mask.repeat(self.out_w_per_mask, 1, self.in_w_per_mask, 1)
        return mask.reshape(self.out_features, self.in_features)

    def produce_mask(self):
        mask = self.mask.produce_mask(self.is_ablate)
        return mask

    def forward(self, input: torch.tensor):
        # "masked_weight = self.produce_mask_reshaped() * self.weight" is equivalent but slower.
        mask = self.produce_mask()
        masked_weight = mask * self.weight.reshape(
            self.out_w_per_mask, self.out_features // self.out_w_per_mask,
            self.in_w_per_mask, self.in_features // self.in_w_per_mask)
        masked_weight = masked_weight.reshape(self.out_features, self.in_features)
        
        act = F.linear(input, masked_weight, self.bias)
        clean_act = F.linear(input, self.weight, self.bias)
        if self.do_caching:
            if self.cached_activation is None:
                self.cached_activation = clean_act.detach()
        return act
    
    def activate_caching(self, caching = True):
        self.cached_activation = None
        self.do_caching = caching

    @classmethod
    def from_layer(cls, layer, out_w_per_mask, in_w_per_mask, is_ablate):
        assert type(layer) == nn.modules.linear.Linear or type(layer) == nn.modules.linear.NonDynamicallyQuantizableLinear
        res = cls(in_features = layer.in_features, out_features = layer.out_features, bias = layer.bias is not None, 
                  out_w_per_mask = out_w_per_mask, in_w_per_mask = in_w_per_mask, is_ablate=is_ablate)
        res.weight = layer.weight
        res.bias = layer.bias
        return res
    
class MaskedMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, mask_p = 0.9, out_w_per_mask = 1, in_w_per_mask = 1, is_ablate=False):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, bias=True, add_zero_attn=False, kdim=None, vdim=None)
        self.num_heads = num_heads
        self.out_w_per_mask = out_w_per_mask
        self.in_w_per_mask = in_w_per_mask
        
        assert embed_dim % num_heads == 0, "{} % {} not 0".format(embed_dim, num_heads)
        mask_dim = (1, embed_dim // out_w_per_mask, 1, 3*embed_dim // in_w_per_mask)
        self.mask = L0Mask(mask_dim)
        self.out_proj = MaskedLinear.from_layer(self.out_proj, out_w_per_mask, in_w_per_mask, is_ablate)
        
        self.cached_activation = None
        self.do_caching = True
        self.is_ablate = is_ablate

    def produce_mask_reshaped(self):
        mask = self.mask.produce_mask(self.is_ablate)
        mask = mask.repeat(self.out_w_per_mask, 1, self.in_w_per_mask, 1)
        return mask.reshape(self.embed_dim, self.embed_dim)

    def produce_mask(self):
        mask = self.mask.produce_mask(self.is_ablate)
        return mask

    def forward(self, query, key, value, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False):
        masked_weight = self.produce_mask() * self.in_proj_weight.reshape(
            self.out_w_per_mask, self.embed_dim // self.out_w_per_mask,
            self.in_w_per_mask, 3*self.embed_dim // self.in_w_per_mask)
        masked_weight = masked_weight.reshape(3*self.embed_dim, self.embed_dim)

        masked_weight_out = self.out_proj.produce_mask() * self.out_proj.weight.reshape(
            self.out_w_per_mask, self.embed_dim // self.out_w_per_mask,
            self.in_w_per_mask, self.embed_dim // self.in_w_per_mask)
        masked_weight_out = masked_weight_out.reshape(self.embed_dim, self.embed_dim)
        act = F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, 
                                            masked_weight, self.in_proj_bias, self.bias_k, self.bias_v, 
                                            self.add_zero_attn, self.dropout, masked_weight_out, 
                                            self.out_proj.bias, self.training, key_padding_mask, need_weights, 
                                            attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)
        clean_act = F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads,
                                            self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v,
                                            self.add_zero_attn, self.dropout, self.out_proj.weight,
                                            self.out_proj.bias, self.training, key_padding_mask, need_weights,
                                            attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if self.do_caching:
            if self.cached_activation is None:
                self.cached_activation = clean_act[0].detach()
        return act
    
    def activate_caching(self, caching = True):
        self.cached_activation = None
        self.do_caching = caching

    @classmethod
    def from_layer(cls, layer, out_w_per_mask, in_w_per_mask, is_ablate):
        assert type(layer) == nn.modules.activation.MultiheadAttention
        res = cls(embed_dim = layer.embed_dim, num_heads = layer.num_heads, 
                  out_w_per_mask = out_w_per_mask, in_w_per_mask = in_w_per_mask, is_ablate=is_ablate)
        res.in_proj_weight = layer.in_proj_weight
        res.in_proj_bias = layer.in_proj_bias
        res.out_proj.weight = layer.out_proj.weight
        res.out_proj.bias = layer.out_proj.bias
        res.bias_k = layer.bias_k
        res.bias_v = layer.bias_v
        res.add_zero_attn = layer.add_zero_attn
        res.dropout = layer.dropout
        res.training = layer.training
        return res
    
