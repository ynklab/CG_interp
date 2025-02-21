from model import Translator, Translator_mask, Translator_mask_probe, Translator_probe
from data import get_data, create_mask, generate_square_subsequent_mask, generate_square_subsequent_mask_batch, gen_split
import torch
import torch.nn as nn
import numpy as np
import random
from argparse import ArgumentParser
from datetime import date

def get_sparsity(model):
    layer_names = ['linear1', 'linear2', 'self_attn', 'multihead_attn']
    parent_types = [nn.TransformerEncoderLayer, nn.TransformerDecoderLayer]
    for name, module in model.named_modules():
        dense_one = 0
        dense_total = 0
        attn_one = 0
        attn_total = 0
        skip = True
        for layer_name in layer_names:
            if hasattr(module, layer_name) and type(module) in parent_types:
                skip = False
                layer = getattr(module, layer_name)
                if 'linear' in layer_name:
                    mask = layer.produce_mask()
                    dense_one += mask.sum()
                    dense_total += mask.numel()

                if 'attn' in layer_name:
                    mask = layer.produce_mask()
                    mask_o = layer.out_proj.produce_mask()
                    attn_one += mask.sum()
                    attn_one += mask_o.sum()
                    attn_total += mask.numel()
                    attn_total += mask_o.numel()
        if skip:
            continue
        print(f'{name} dense sparsity: {dense_one/dense_total}')
        print(f'{name} attn sparsity: {attn_one/attn_total}')
        print(f'{name} total sparsity: {(dense_one + attn_one)/(dense_total + attn_total)}')

        
def main(opts):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_split(opts)
    test_dl, gen_dl, src_vocab, tgt_vocab, src_transform, _, special_symbols = get_data(opts, eval=True)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    model = Translator_mask(
                num_encoder_layers=opts.enc_layers,
                num_decoder_layers=opts.dec_layers,
                embed_size=opts.embed_size,
                num_heads=opts.attn_heads,
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                dim_feedforward=opts.dim_feedforward,
                dropout=opts.dropout,
                out_w_per_mask=1,
                in_w_per_mask=1,
                is_ablate=opts.ablate,
                embed_init=opts.embed_init,
                embed_scale=opts.embed_scale
            ).to(DEVICE)
    model.load_state_dict(torch.load(opts.model_path), strict=False)
    print('start counting...')
    get_sparsity(model)

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Machine Translator training and inference",
    )

    # Inference mode
    parser.add_argument("--inference", action="store_true",
                        help="Set true to run inference")
    parser.add_argument("--model_path", type=str,
                        help="Path to the model to run inference on")
    parser.add_argument("--beam", action="store_true",
                        help="Set true to run beam search")
    parser.add_argument("--reset", action="store_true",
                        help="Set true to reset the model weights")
    
    # Pruning mode
    parser.add_argument("--prune", action="store_true",
                        help="Set true to run pruning")
    parser.add_argument("--ablate", action="store_true",
                        help="Set true to reverse mask", default=False)
    parser.add_argument("--lambda_init", type=float, default=1,)
    parser.add_argument("--lambda_final", type=float, default=1,)
    parser.add_argument("--lambda_startup_frac", type=float, default=0.1,)
    parser.add_argument("--lambda_warmup_frac", type=float, default=0.5,)
    parser.add_argument("--lr_base", type=float, default=5e-5,)
    parser.add_argument("--mask_lr_base", type=float, default=0.1,)
    parser.add_argument("--lr_warmup_frac", type=float, default=0.1,)

    # Translation settings
    parser.add_argument("--src", type=str, default="en",
                        help="Source language (translating FROM this language)")
    parser.add_argument("--tgt", type=str, default="j",
                        help="Target language (translating TO this language)")
    parser.add_argument("--data_path", type=str, default="../../../data/restrict",
                        help="Path to the data")
    parser.add_argument("--data_train_path", type=str, default="../../../data/restrict/train",)
    parser.add_argument("--result_path", type=str, default="../../../results",
                        help="Path to the results")

    # Training settings
    parser.add_argument("--pos_enc", type=str, default="default",)
    parser.add_argument("--hint", type=str, default="default",
                        help="Set true to run hint")
    parser.add_argument("-e", "--epochs", type=int, default=30,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Default learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--backend", type=str, default="gpu",
                        help="Batch size")
    parser.add_argument("--probe", action="store_true",
                        help="Set true to run probing")
    parser.add_argument("--counterfactual", action="store_true",
                        help="Set true to run counterfactual")
    parser.add_argument("--is_masked", action="store_true",
                        help="Set true to run masked")
    
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=6,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=6,
                        help="Number of decoder layers")
    parser.add_argument("--embed_size", type=int, default=512,
                        help="Size of the language embedding")
    parser.add_argument("--dim_feedforward", type=int, default=2048,
                        help="Feedforward dimensionality")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout")
    parser.add_argument("--embed_init", type=str, default="xavier",
                        help="How to initialize the embeddings")
    parser.add_argument("--embed_scale", type=str, default="default",
                        help="How to scale the embeddings")

    # Logging settings
    parser.add_argument("--logging_dir", type=str, default="./" + str(date.today()) + "/",
                        help="Where the output of this program should be placed")

    # Just for continuous integration
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scrub", action="store_true")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)