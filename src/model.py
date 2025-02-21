# https://github.com/pytorch/examples/blob/main/language_translation/src/model.py

import math

import torch
from torch.nn import functional as F
from torch import nn
from masked_linear import MaskedLinear, MaskedMultiheadAttention
from probe_layer import ProbedTransformerEncoderLayer
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size,
        emb_scale,
        dropout,
        maxlen=5000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        if emb_scale == 'down':
            pos_embedding = pos_embedding / math.sqrt(emb_size)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding, emb_size, emb_scale):
        if emb_scale == 'up':
            return self.dropout(token_embedding * math.sqrt(emb_size)  + self.pos_embedding[:token_embedding.size(0), :])
        else:
            return self.dropout(token_embedding  + self.pos_embedding[:token_embedding.size(0), :])

class Translator(nn.Module):
    def __init__(
            self,
            num_encoder_layers,
            num_decoder_layers,
            embed_size,
            num_heads,
            src_vocab_size,
            tgt_vocab_size,
            dim_feedforward,
            dropout,
            embed_init,
            embed_scale
        ):
        super(Translator, self).__init__()

        # Output of embedding must be equal (embed_size)
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)

        self.pos_enc = PositionalEncoding(embed_size, embed_scale, dropout)
        self.embed_scale = embed_scale
        self.embed_size = embed_size
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.ff = nn.Linear(embed_size, tgt_vocab_size)

        self._init_weights(embed_init)

    def _init_weights(self, embed_init):
        for pn, p in self.named_parameters():
            if p.dim() > 1:
                if embed_init == 'xavier':
                    if 'weight' in pn:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.zeros_(p)
                elif embed_init == 'kaiming':
                    if 'weight' in pn:
                        nn.init.kaiming_uniform_(p)
                    else:
                        nn.init.zeros_(p)
                else:
                    raise ValueError(f"Unknown initialization method: {embed_init}")

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):

        src_emb = self.pos_enc(self.src_embedding(src), self.embed_size, self.embed_scale)
        tgt_emb = self.pos_enc(self.tgt_embedding(trg), self.embed_size, self.embed_scale)

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask
        )

        return self.ff(outs)

    def encode(self, src, src_mask, src_padding_mask=None):

        embed = self.src_embedding(src)

        pos_enc = self.pos_enc(embed, self.embed_size, self.embed_scale)

        return self.transformer.encoder(pos_enc, src_mask, src_padding_mask)

    def decode(self, tgt, memory, tgt_mask, memory_mask=None,tgt_padding_mask=None):
        
        embed = self.tgt_embedding(tgt)

        pos_enc = self.pos_enc(embed, self.embed_size, self.embed_scale)

        return self.transformer.decoder(pos_enc, memory, tgt_mask, memory_key_padding_mask=memory_mask, tgt_key_padding_mask=tgt_padding_mask)
    
    def get_layers(self):
        encoder_layers = []
        decoder_layers = []
        for module in self.modules():
            if isinstance(module, (nn.TransformerEncoderLayer)):
                encoder_layers.append(module)
            elif isinstance(module, (nn.TransformerDecoderLayer)):
                decoder_layers.append(module)
        return encoder_layers, decoder_layers

    
class Translator_mask(Translator):
    def __init__(self,*args, out_w_per_mask, in_w_per_mask, is_ablate, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_layers_with_masked(out_w_per_mask, in_w_per_mask, is_ablate)

        if torch.cuda.is_available():
            self.cuda()

    def replace_layers_with_masked(self, out_w_per_mask, in_w_per_mask, is_ablate, verbose = False):
        """
        Replaces layers with their masked versions.
        out_w_per_mask: the number of output dims covered by a single mask parameter
        in_w_per_mask: the same as above but for input dims
        
        ex: (1,1) for weight masking
            (768,1) for neuron masking
            (768, 768) for layer masking

        Transformer:
        transformer.{encoder, decoder}.layers.{0-6}.self_attn.out_proj
        transformer.decoder.layers.{0-6}.multihead_attn.out_proj
        transformer_{encoder, decoder}.layers.{0-6}.linear{1-2}
        ff

        """
        def replace_layers(layer_names, parent_types, replacement_linear, replacement_attn):
            for module_name, module in self.named_modules():
                for layer_name in layer_names:
                    if hasattr(module, layer_name) and type(module) in parent_types:
                        layer = getattr(module, layer_name)
                        if 'linear' in layer_name:
                            setattr(module, layer_name, replacement_linear(layer))
                        elif 'attn' in layer_name:
                            setattr(module, layer_name, replacement_attn(layer))
                        else:
                            raise ValueError(f"Unknown layer type: {layer_name}")
                        if verbose:
                            print("Replaced {} in {}".format(layer_name, module_name))

        replace_layers(('linear1', 'linear2', 'self_attn', 'multihead_attn'), (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer),
                       lambda x: MaskedLinear.from_layer(x, out_w_per_mask, in_w_per_mask, is_ablate),
                       lambda x: MaskedMultiheadAttention.from_layer(x, out_w_per_mask, in_w_per_mask, is_ablate))
        
    def compute_total_regularizer(self):
        total, n = 0, 0
        for module in self.modules():
            if hasattr(module, 'regularizer'):
                total += module.regularizer()
                n += 1
        return total / n
    
    def freeze(self):
        '''
        freeze all layers of the transformer model
        '''
        for name, parameter in self.named_parameters():
            if 'mask_scores' not in name:
                parameter.requires_grad = False

    def reset_weights(self):
        for _, module in self.named_modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def reset_cache(self):
        for _, module in self.named_modules():
            if hasattr(module, 'activate_caching'):
                module.activate_caching(True)

    def get_layers(self):
        encoder_layers = []
        decoder_layers = []
        for module in self.modules():
            if isinstance(module, (nn.TransformerEncoderLayer)):
                encoder_layers.append(module)
            elif isinstance(module, (nn.TransformerDecoderLayer)):
                decoder_layers.append(module)
        return encoder_layers, decoder_layers

class Probe_Classifier(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(Probe_Classifier, self).__init__()
        self.out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        return self.out(x)
    
class Translator_probe(Translator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_layers_with_probe()

        if torch.cuda.is_available():
            self.cuda()

    def replace_layers_with_probe(self, verbose = False):
        """
        Replace the last layer of the encoder with a probe classifier
        """
        def replace_layers(layer_names, parent_types, replacement):
            for module_name, module in self.named_modules():
                for layer_name in layer_names:
                    if hasattr(module, layer_name) and type(module) in parent_types:
                        module_list = getattr(module, layer_name)
                        new_module_list = nn.ModuleList()
                        for layer in module_list:
                            new_module_list.append(replacement(layer))
                        
                        setattr(module, layer_name, new_module_list)
                        if verbose:
                            print("Replaced {} in {}".format(layer_name, module_name))

        replace_layers(('layers',), (nn.TransformerEncoder,), lambda x: ProbedTransformerEncoderLayer.from_layer(x))
    
    def reset_cache(self):
        for _, module in self.named_modules():
            if hasattr(module, 'activate_caching'):
                module.activate_caching()

    def get_activations(self):
        activations = []
        for module_name, module in self.named_modules():
            if hasattr(module, 'get_cached_activation'):
                activations.append(module.get_cached_activation())
        return activations
    
    def get_params(self, idx_layer):
        params = set()
        for module_name, module in self.named_modules():
            for name, param in module.named_parameters():
                full_param_name = f"{module_name}.{name}" if module_name else name
                if hasattr(module, 'get_cached_activation') and str(idx_layer) in full_param_name:
                    param.requires_grad = True
                    params.add(full_param_name)
        param_dict = {name: param for name, param in self.named_parameters()}
        return [param_dict[name] for name in sorted(list(params))]

class Translator_mask_probe(Translator_mask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_layers_with_probe()

        if torch.cuda.is_available():
            self.cuda()

    def replace_layers_with_probe(self, verbose = False):
        """
        Replace the last layer of the encoder with a probe classifier
        """
        def replace_layers(layer_names, parent_types, replacement):
            for module_name, module in self.named_modules():
                for layer_name in layer_names:
                    if hasattr(module, layer_name) and type(module) in parent_types:
                        module_list = getattr(module, layer_name)
                        new_module_list = nn.ModuleList()
                        for layer in module_list:
                            new_module_list.append(replacement(layer))
                        
                        setattr(module, layer_name, new_module_list)
                        if verbose:
                            print("Replaced {} in {}".format(layer_name, module_name))

        replace_layers(('layers',), (nn.TransformerEncoder,), lambda x: ProbedTransformerEncoderLayer.from_layer(x))
    
    def reset_cache(self):
        for _, module in self.named_modules():
            if hasattr(module, 'activate_caching'):
                module.activate_caching()

    def get_activations(self):
        activations = []
        for module_name, module in self.named_modules():
            if hasattr(module, 'get_cached_activation'):
                activations.append(module.get_cached_activation())
        return activations
    
    def get_params(self, idx_layer):
        str_idx_layer = f'.{str(idx_layer)}.'
        params = set()
        for module_name, module in self.named_modules():
            for name, param in module.named_parameters():
                full_param_name = f"{module_name}.{name}" if module_name else name
                if hasattr(module, 'get_cached_activation') and str_idx_layer in full_param_name:
                    param.requires_grad = True
                    params.add(full_param_name)
        param_dict = {name: param for name, param in self.named_parameters()}
        return [param_dict[name] for name in sorted(list(params))]
