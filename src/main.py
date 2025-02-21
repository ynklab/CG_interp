# https://github.com/pytorch/examples/blob/main/language_translation/main.py
# https://github.com/jarobyte91/pytorch_beam_search/blob/master/src/pytorch_beam_search/seq2seq/search_algorithms.py
from time import time # Track how long an epoch takes
import os # Creating and finding files/directories
import logging # Logging tools
from datetime import date # Logging the date for model versioning

import torch # For ML
from tqdm import tqdm # For fancy progress bars

from model import Translator, Translator_mask, Translator_mask_probe, Translator_probe
from data import get_data, create_mask, generate_square_subsequent_mask, generate_square_subsequent_mask_batch, gen_split # Loading data and data preprocessing
from argparse import ArgumentParser # For args
import torch.nn as nn
import gc
import torch.utils.data as tud
import random
import numpy as np

def beam_decode(model, src, src_mask, src_padding_mask, tgt_vocab, max_len, start_symbol, end_symbol, pad_symbol, opts, device):
    batch_size = src.shape[1]
    beam_width = 5

    src = src.to(device)
    src_mask = src_mask.to(device)
    
    # Encode input
    memory = model.encode(src, src_mask, src_padding_mask)

    # Output will be stored here
    ys = torch.ones(1, batch_size, 1).fill_(start_symbol).type(torch.long).repeat((1, 1, beam_width)).to(device)

    probs = torch.zeros(batch_size, beam_width).to(device)
    translations = [None] * batch_size
    for i in range(max_len):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask_batch(ys.size(0), batch_size, opts, device).type(torch.bool)).to(device)
        
        next_probs = []
        next_chars = []
        for idx in range(beam_width):
            y = ys[:,:,idx]
            tgt_padding_mask = (y == pad_symbol).transpose(0, 1)
            out = model.decode(y, memory, tgt_mask, src_padding_mask, tgt_padding_mask)
            out = out.transpose(0, 1)
            prob = model.ff(out[:, -1, :])
            next_probs.append(prob)
        next_probs = torch.stack(next_probs, axis = 0)
        if i == 0:
            probs = next_probs[0]
            vocabulary_size = next_probs.shape[2]
            probs, next_chars = probs.softmax(-1).topk(k = beam_width, axis = -1)
            ys = torch.cat((ys, next_chars.unsqueeze(0)), axis = 0)
            continue
        else:
            probs = probs.unsqueeze(2).repeat((1, 1, vocabulary_size))
            next_probs = next_probs.transpose(0, 1)
            probs = torch.mul(probs, next_probs).flatten(start_dim=1)
        probs, idx = probs.softmax(-1).topk(k = beam_width, axis = -1)
        next_chars = torch.remainder(idx, vocabulary_size).unsqueeze(0).to(device)
        is_end_symbol = next_chars[0, :, 0] == end_symbol
        best_candidates = idx // vocabulary_size

        prev_chars = ys[-1, :, :]
        for j in range(batch_size):
            if prev_chars[j, 0] == end_symbol:
                prev_chars[j] = torch.tensor([end_symbol]).to(device)
                next_chars[0][j] = torch.tensor([pad_symbol]).to(device)
            elif prev_chars[j, 0] == pad_symbol:
                next_chars[0][j] = torch.tensor([pad_symbol]).to(device)
        prev_chars = prev_chars.gather(1, best_candidates)
        prev_chars = prev_chars.unsqueeze(0).to(device)
        ys = torch.cat((ys[:-1, :, :], prev_chars), axis = 0)
        ys = torch.cat((ys, next_chars), axis = 0)
        for j in range(batch_size):
            if is_end_symbol[j] and translations[j] is None:
                max_prob, idx = torch.max(probs[j], 0)
                translations[j] = list(ys[:, j, 0].cpu().numpy())
        is_all_finished = translations.count(None) == 0
        if is_all_finished:
            break
        torch.cuda.empty_cache()
        gc.collect()

    while None in translations:
        idx = translations.index(None)
        translations[idx] = list(ys[:, idx, 0].cpu().numpy())
    return translations

# Opens an user interface where users can translate an arbitrary sentence
def inference(opts):

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    gen_split(opts)
    test_dl, gen_dl, src_vocab, tgt_vocab, src_transform, _, special_symbols = get_data(opts, eval=True)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Create model
    if opts.prune:
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
        # model.freeze()
    elif opts.counterfactual:
        if opts.is_masked:
            model = Translator_mask_probe(
                num_encoder_layers=opts.enc_layers, 
                num_decoder_layers=opts.dec_layers, 
                num_heads=opts.attn_heads, 
                embed_size=opts.embed_size, 
                dim_feedforward=opts.dim_feedforward, 
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                dropout=opts.dropout, 
                embed_init=opts.embed_init, 
                embed_scale=opts.embed_scale,
                out_w_per_mask=1,
                in_w_per_mask=1,
                is_ablate=False
            ).to(DEVICE)
        else:
            model = Translator_probe(
            num_encoder_layers=opts.enc_layers, 
            num_decoder_layers=opts.dec_layers, 
            num_heads=opts.attn_heads, 
            embed_size=opts.embed_size, 
            dim_feedforward=opts.dim_feedforward, 
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            dropout=opts.dropout, 
            embed_init=opts.embed_init, 
            embed_scale=opts.embed_scale
        ).to(DEVICE)
        model.load_state_dict(torch.load(opts.model_path))
    else:
        model = Translator(
            num_encoder_layers=opts.enc_layers,
            num_decoder_layers=opts.dec_layers,
            embed_size=opts.embed_size,
            num_heads=opts.attn_heads,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            dim_feedforward=opts.dim_feedforward,
            dropout=opts.dropout,
            embed_init=opts.embed_init,
            embed_scale=opts.embed_scale
        ).to(DEVICE)

        # Load in weights
        model.load_state_dict(torch.load(opts.model_path))

    # Set to inference
    model.eval()
    # first eval on test set
    test_preds = []
    for src, tgt in test_dl:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask = (torch.zeros(opts.attn_heads*src.shape[1], src.shape[0], src.shape[0])).type(torch.bool)
        # src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool)
        src_padding_mask = (src == special_symbols["<pad>"]).transpose(0, 1)
        with torch.no_grad():
            tgt_tokens = beam_decode(
                model, src, src_mask, src_padding_mask, tgt_vocab, max_len=256, start_symbol=special_symbols["<bos>"], end_symbol=special_symbols["<eos>"], pad_symbol=special_symbols["<pad>"], opts=opts, device=DEVICE
            )
        output_as_list = tgt_tokens
        output_list_words = [tgt_vocab.lookup_tokens(i) for i in output_as_list]

        # Remove special tokens and convert to string
        translations = [" ".join(words).replace("<bos>", "").replace("<eos>", "").replace("<pad>", "") for words in output_list_words]

        test_preds.extend(translations)
        if opts.dry_run:
            break
    
    with open(f"{opts.result_path}/pred_test.txt", "w") as f:
        f.write("\n".join(test_preds))
    print("test set done")
    # then eval on generated set
    gen_preds = []
    for src, tgt in gen_dl:
        if opts.dry_run:
            break
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask = (torch.zeros(opts.attn_heads*src.shape[1], src.shape[0], src.shape[0])).type(torch.bool)
        src_padding_mask = (src == special_symbols["<pad>"]).transpose(0, 1)
        # Decode
        with torch.no_grad():
            tgt_tokens = beam_decode(
                model, src, src_mask, src_padding_mask, tgt_vocab, max_len=256, start_symbol=special_symbols["<bos>"], end_symbol=special_symbols["<eos>"], pad_symbol=special_symbols["<pad>"], opts=opts, device=DEVICE
            )
            output_as_list = tgt_tokens
            output_list_words = [tgt_vocab.lookup_tokens(i) for i in output_as_list]

            # Remove special tokens and convert to string
            translations = [" ".join(words).replace("<bos>", "").replace("<eos>", "") for words in output_list_words]

            gen_preds.extend(translations)
    # write to file
    with open(f"{opts.result_path}/pred_gen.txt", "w") as f:
        f.write("\n".join(gen_preds))

# Train the model for 1 epoch
def train(model, ref_model, train_dl, id_train_dl, loss_fn, optim, special_symbols, processed, set_lr, opts):

    # Object for accumulating losses
    losses = 0
    model.train()
    for i, batch in enumerate(train_dl):
        src, tgt = batch
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_tgt = src
        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_tgt, tgt_input, special_symbols["<pad>"], DEVICE)
        
        if opts.prune:
            logits = model(src_tgt, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            src_ja = src
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_ja, tgt_input, special_symbols["<pad>"], DEVICE)
        else:
            logits = model(src_tgt, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        optim.zero_grad()

        tgt_out = tgt[1:, :]
        if opts.prune:
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss += opts.lambda_init * model.compute_total_regularizer()
        else:
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optim.step()
        losses += loss.item()
        if opts.dry_run:
            break
    return losses / len(list(train_dl))

# Check the model accuracy on the validation dataset
def validate(model, ref_model, valid_dl, id_valid_dl, loss_fn, special_symbols, opts):
    
    losses = 0
    model.eval()
    for i, batch in enumerate(valid_dl):
        src, tgt = batch
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)
        _ = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(list(valid_dl))

# Train the model
def main(opts):

    # Set up logging
    os.makedirs(opts.logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=opts.logging_dir + "log.txt", level=logging.INFO)

    # This prints it to the screen as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info(f"Translation task: {opts.src} -> {opts.tgt}")
    logging.info(f"Using device: {DEVICE}")

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    gen_split(opts)
    id_train_dl = None
    id_valid_dl = None
    if opts.prune:
        train_dl, valid_dl, id_valid_dl, src_vocab, tgt_vocab, src_transform, tgt_transform, special_symbols = get_data(opts)
    else:
        train_dl, valid_dl, src_vocab, tgt_vocab, _, _, special_symbols = get_data(opts)

    logging.info("Loaded data")

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    logging.info(f"{opts.src} vocab size: {src_vocab_size}")
    logging.info(f"{opts.tgt} vocab size: {tgt_vocab_size}")

    # Create model
    model = None
    if opts.prune:
        model = Translator_mask(
            num_encoder_layers=opts.enc_layers,
            num_decoder_layers=opts.dec_layers,
            embed_size=opts.embed_size,
            num_heads=opts.attn_heads,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            dim_feedforward=opts.dim_feedforward,
            dropout=opts.dropout,
            embed_init=opts.embed_init,
            embed_scale=opts.embed_scale,
            out_w_per_mask=1,
            in_w_per_mask=1,
            is_ablate=opts.ablate
        ).to(DEVICE)
        model.load_state_dict(torch.load(opts.model_path), strict=False)
        model.freeze()
    else:
        model = Translator(
            num_encoder_layers=opts.enc_layers,
            num_decoder_layers=opts.dec_layers,
            embed_size=opts.embed_size,
            num_heads=opts.attn_heads,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            dim_feedforward=opts.dim_feedforward,
            dropout=opts.dropout,
            embed_init=opts.embed_init,
            embed_scale=opts.embed_scale
        ).to(DEVICE)
    
    if opts.reset:
        model.reset_weights()
    
    logging.info("Model created... starting training!")

    # Set up our learning tools
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    # These special values are from the "Attention is all you need" paper
    optim = None
    if opts.prune:
        params = [p for n, p in model.named_parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr = opts.lr)
    else:
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        decay = set()
        no_decay = set()
        for mname, m in model.named_modules():
            for pname, p in m.named_parameters():
                full_name = f"{mname}.{pname}" if mname else pname
                if pname.endswith('bias'):
                    no_decay.add(full_name)
                elif pname.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(full_name)
                elif pname.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(full_name)
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": opts.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optim = torch.optim.AdamW(optim_groups, lr=opts.lr)
        # increase learning rate from 0 to lr over the first 10% of training
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.0001, total_iters=20)

    def set_lr(lr_ratio):
            for param_group in optim.param_groups:
                param_group['lr'] = param_group['lr_base'] * lr_ratio

    best_val_loss = 1e6
    processed = 0
    ref_model = model
    for idx, epoch in enumerate(range(1, opts.epochs+1)):

        train_loss = train(model, ref_model, train_dl, id_train_dl, loss_fn, optim, special_symbols, processed, set_lr, opts)
        val_loss   = validate(model, ref_model, valid_dl, id_valid_dl, loss_fn, special_symbols, opts)
        if not opts.prune:
            scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # logging.info("New best model, saving...")
            torch.save(model.state_dict(), opts.logging_dir + "best.pt")
        if epoch % 20 == 0:
            torch.save(model.state_dict(), opts.logging_dir + f"epoch_{epoch}.pt")

        logger.info(f"Epoch: {epoch}\tTrain loss: {train_loss:.5f}\tVal loss: {val_loss:.5f}")
        if opts.dry_run:
            break

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
    if args.inference:
        inference(args)
    else:
        main(args)