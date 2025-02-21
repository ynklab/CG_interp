import torch
import argparse
from model import Probe_Classifier, Translator, Translator_mask, Translator_mask_probe, Translator_probe
from data import create_mask, get_data, gen_split
import numpy as np
import random
import logging
import os

def probe_train(classifier, model, train_dl, loss_fn, optim, special_symbols, opts, device, idx_layer):
    classifier.train()
    model.eval()
    losses = 0
    count = 0
    for batch in train_dl:
        # use label tgt for input tgt because it is not used in encoder anyway.
        src, tgt = batch
        src = src.to(device)
        src_mask = (torch.zeros(opts.attn_heads*src.shape[1], src.shape[0], src.shape[0])).type(torch.bool).to(device)
        src_padding_mask = (src == special_symbols["<pad>"]).transpose(0, 1).to(device)
        tgt = tgt.to(device)
        with torch.no_grad():
            _ = model.encode(src, src_mask, src_padding_mask)
        
        activation = model.get_activations()[idx_layer]
        
        if 'leace' in opts.probe_mode:
            out = classifier(activation)
            out = torch.transpose(out, 0, 1)
            out = torch.transpose(out, 1, 2)
            tgt = torch.transpose(tgt, 0, 1)
            tgt = torch.transpose(tgt, 1, 2)
            tgt = tgt.float()
            loss = loss_fn(out, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses += loss.item()
            model.reset_cache()
            count += activation.shape[1]
        elif 'translation' in opts.probe_mode:
            out = classifier(activation)
            out = torch.transpose(out, 0, 1)
            out = torch.transpose(out, 1, 2)
            tgt = torch.transpose(tgt, 0, 1)
            loss = loss_fn(out, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses += loss.item()
            model.reset_cache()
            count += activation.shape[1]
        else:
            raise ValueError('Invalid probe mode')
    
    return losses / count

def probe_val(classifier, model, val_dl, loss_fn, special_symbols, opts, device, idx_layer):
    classifier.eval()
    model.eval()
    losses = 0
    count = 0
    for batch in val_dl:
        src, tgt = batch
        src = src.to(device)
        src_mask = (torch.zeros(opts.attn_heads*src.shape[1], src.shape[0], src.shape[0])).type(torch.bool).to(device)
        src_padding_mask = (src == special_symbols["<pad>"]).transpose(0, 1).to(device)
        tgt = tgt.to(device)
        with torch.no_grad():
            _ = model.encode(src, src_mask, src_padding_mask)

        activation = model.get_activations()[idx_layer]
        if 'leace' in opts.probe_mode:
            out = classifier(activation)
            out = torch.transpose(out, 0, 1)
            out = torch.transpose(out, 1, 2)
            tgt = torch.transpose(tgt, 0, 1)
            tgt = torch.transpose(tgt, 1, 2)
            tgt = tgt.float()
            loss = loss_fn(out, tgt)
            loss.backward()
            losses += loss.item()
            model.reset_cache()
            count += activation.shape[1]
        elif 'translation' in opts.probe_mode:
            out = classifier(activation)
            out = torch.transpose(out, 0, 1)
            out = torch.transpose(out, 1, 2)
            tgt = torch.transpose(tgt, 0, 1)
            loss = loss_fn(out, tgt)
            losses += loss.item()
            count += activation.shape[1]
            model.reset_cache()
        else:
            raise ValueError('Invalid probe mode')

    return losses / count

def probe_test(classifier, model, test_dl, tgt_vocab, special_symbols, device, opts):
    test_results = []
    for batch in test_dl:
        src, tgt = batch
        src = src.to(device)
        src_mask = (torch.zeros(opts.attn_heads*src.shape[1], src.shape[0], src.shape[0])).type(torch.bool).to(device)
        src_padding_mask = (src == special_symbols["<pad>"]).transpose(0, 1).to(device)
        tgt = tgt.to(device)
        with torch.no_grad():
            _ = model.encode(src, src_mask, src_padding_mask)
        
        activation = model.get_activations()[opts.probe_layer]
        if 'leace' in opts.probe_mode:
            out = classifier(activation)
            softmax = torch.nn.Sigmoid()
            out = softmax(out)
            out = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
            out = out.transpose(0, 1)
            out = out.int()
            out = out.tolist()
            for i in range(len(out)):
                out_i = ''
                for j, idx in enumerate(out[i]):
                    if src_padding_mask[i][j] == 0:
                        out_i += str(idx) + '\t'
                out_i = out_i[:-1]
                test_results.append(out_i)
            model.reset_cache()
        elif 'translation' in opts.probe_mode:
            out = classifier(activation)
            softmax = torch.nn.Softmax(dim=-1)
            out = softmax(out)
            out = out.argmax(dim=-1)
            out = out.transpose(0, 1)
            out = out.tolist()
            out = [tgt_vocab.lookup_tokens(idx) for idx in out]
            out = [' '.join(out_i) for out_i in out]
            test_results.extend(out)
            model.reset_cache()
        else:
            raise ValueError('Invalid probe mode')
        model.reset_cache()

    return test_results

def inference(opts):
    test_dl, gen_dl, src_vocab, tgt_vocab, tgt_vocab_,  _, _, special_symbols = get_data(opts, eval=True)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
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
        model.load_state_dict(torch.load(opts.model_path))
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
    num_classes = 0
    embed_size = opts.embed_size
    if 'dependency' in opts.probe_mode:
        num_classes = 12
    elif 'constituency' in opts.probe_mode:
        num_classes = 20
    elif 'translation' in opts.probe_mode:
        num_classes = len(tgt_vocab_)
    classifier = Probe_Classifier(
        embed_size=embed_size,
        num_classes=num_classes
    ).to(DEVICE)
    classifier.load_state_dict(torch.load(opts.classifier_path))
    model.eval()
    classifier.eval()
    test_results = probe_test(classifier, model, test_dl, tgt_vocab_, special_symbols, DEVICE, opts)
    with open(f'{opts.result_path}/pred_test.txt', 'w') as f:
        for res in test_results:
            f.write(res + '\n')

    gen_results = probe_test(classifier, model, gen_dl, tgt_vocab_, special_symbols, DEVICE, opts)
    with open(f'{opts.result_path}/pred_gen.txt', 'w') as f:
        for res in gen_results:
            f.write(res + '\n')

def main(opts):
    os.makedirs(opts.logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=opts.logging_dir + "log.txt", level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    train_dl, val_dl, src_vocab, tgt_vocab, tgt_vocab_, _, _, special_symbols = get_data(opts, eval=False)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
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
    
    model.load_state_dict(torch.load(opts.model_path), strict=False)
    num_classes = 0
    embed_size = opts.embed_size
    if 'dependency' in opts.probe_mode:
        num_classes = 12
    elif 'constituency' in opts.probe_mode:
        num_classes = 20
    elif 'translation' in opts.probe_mode:
        num_classes = len(tgt_vocab_)
    classifier = Probe_Classifier(
        embed_size=embed_size,
        num_classes=num_classes
    ).to(DEVICE)
    if 'translation' in opts.probe_mode:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(classifier.parameters(), lr=opts.lr)
    for epoch in range(opts.epochs):
        train_loss = probe_train(classifier, model, train_dl, loss_fn, optim, special_symbols, opts, DEVICE, opts.probe_layer)
        val_loss = probe_val(classifier, model, val_dl, loss_fn, special_symbols, opts, DEVICE, opts.probe_layer)
        
        logger.info(f'Epoch {epoch+1}/{opts.epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}')
        if (epoch + 1) % 10 == 0:
            torch.save(classifier.state_dict(), opts.logging_dir + f'epoch_{epoch+1}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model')
    parser.add_argument('--src', type=str, help='source language')
    parser.add_argument('--tgt', type=str, help='target language')
    parser.add_argument("--prune", action="store_true",
                        help="Set true to run pruning")
    parser.add_argument("--ablate", action="store_true",
                        help="Set true to reverse mask", default=False)
    parser.add_argument("--beam", action="store_true",
                        help="Set true to run beam search")
    parser.add_argument("--reset", action="store_true",
                        help="Set true to reset the model weights")
    parser.add_argument('--inference', action='store_true', help='whether to run inference')
    parser.add_argument('--classifier_path', type=str, help='path to the classifier')
    parser.add_argument('--is_masked', action='store_true', help='whether the model is masked')
    parser.add_argument('--probe', action='store_true', help='whether to probe')
    parser.add_argument('--probe_hint', action='store_true', help='whether to probe with hint')
    parser.add_argument('--probe_layer', type=int, help='layer to probe')
    parser.add_argument('--probe_mode', type=str, help='mode of span', default='iobj')
    parser.add_argument("--pos_enc", type=str, default="default",)
    parser.add_argument("--hint", type=str, default="default",
                        help="Set true to run hint")
    parser.add_argument("--backend", type=str, default="gpu",
                        help="Batch size")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--data_path', type=str, help='path to the data')
    parser.add_argument("--data_train_path", type=str, default="../../../data/restrict/train",)
    parser.add_argument('--span_data_path', type=str, help='path to the span data')
    parser.add_argument('--result_path', type=str, help='path to the result')
    parser.add_argument('--attn_heads', type=int, help='number of attention heads')
    parser.add_argument('--enc_layers', type=int, help='number of encoder layers')
    parser.add_argument('--dec_layers', type=int, help='number of decoder layers')
    parser.add_argument('--embed_size', type=int, default=512, help='embedding dimension')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--embed_init', type=str, help='embedding initialization')
    parser.add_argument('--embed_scale', type=str, help='embedding scale')
    parser.add_argument('--logging_dir', type=str, help='logging directory')
    parser.add_argument('--scrub', action='store_true', help='whether to scrub')

    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.inference:
        inference(args)
    else:
        main(args)




