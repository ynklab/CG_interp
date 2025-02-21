# https://github.com/pytorch/examples/blob/main/language_translation/src/data.py

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchdata import datapipes
import ast
from torch.utils.data import Dataset

def _yield_tokens(iterable_data, tokenizer, src):

    # Iterable data stores the samples as (src, tgt) so this will help us select just one language or the other
    index = 0 if src else 1

    for data in iterable_data:
        yield tokenizer(data)

# Get data, tokenizer, text transform, vocab objs, etc. Everything we
# need to start training the model
        
class SGETDataset(Dataset):
    def __init__(self, path_src, path_tgt):
        with open(path_src, 'r') as f:
            self.sentences_src = f.readlines()
        with open(path_tgt, 'r') as f:
            self.sentences_tgt = f.readlines()
    
    def __len__(self):
        assert len(self.sentences_src) == len(self.sentences_tgt)
        return len(self.sentences_src)
    
    def __getitem__(self, idx):
        return self.sentences_src[idx], self.sentences_tgt[idx]

def get_data(opts, eval=False, skip=False):

    src_lang = opts.src #en
    tgt_lang = opts.tgt #ja, sem (semantic parsing), sym (syntactic parsing), pos (pos tagging)

    # Define a token "unkown", "padding", "beginning of sentence", and "end of sentence"
    special_symbols = {
        "<unk>":0,
        "<pad>":1,
        "<bos>":2,
        "<eos>":3,
    }

    src_tokenizer = get_tokenizer(tokenizer=None)
    tgt_tokenizer = get_tokenizer(tokenizer=None)
    if "hint" in opts.hint:
        train_ja_src_path = f"{opts.data_train_path}/train_{src_lang}_hint.txt"
        if tgt_lang == 'ja':
            train_ja_tgt_path = f"{opts.data_train_path}/train_ja_hint.txt"
            train_cogs_tgt_path = f"{opts.data_train_path}/train_lf_hint.txt"
        elif tgt_lang == 'cogs':
            train_ja_tgt_path = f"{opts.data_train_path}/train_ja_hint.txt"
            train_cogs_tgt_path = f"{opts.data_train_path}/train_lf_hint.txt"
    else:
        train_ja_src_path = f"{opts.data_train_path}/train_{src_lang}.txt"
        train_ja_tgt_path = f"{opts.data_train_path}/train_ja.txt"
        train_cogs_tgt_path = f"{opts.data_train_path}/train_lf.txt"

    dev_ja_src_path = f"{opts.data_path}/gen_{src_lang}.txt"
    dev_ja_tgt_path = f"{opts.data_path}/gen_ja.txt"
    dev_cogs_tgt_path = f"{opts.data_path}/gen_lf.txt"

    train_iterator_ja_src = datapipes.iter.FileOpener([train_ja_src_path], mode='rt').readlines(return_path=False)
    dev_iterator_ja_src = datapipes.iter.FileOpener([dev_ja_src_path], mode='rt').readlines(return_path=False)
    train_iterator_ja_tgt = datapipes.iter.FileOpener([train_ja_tgt_path], mode='rt').readlines(return_path=False)
    dev_iterator_ja_tgt = datapipes.iter.FileOpener([dev_ja_tgt_path], mode='rt').readlines(return_path=False)
    train_iterator_cogs_tgt = datapipes.iter.FileOpener([train_cogs_tgt_path], mode='rt').readlines(return_path=False)
    dev_iterator_cogs_tgt = datapipes.iter.FileOpener([dev_cogs_tgt_path], mode='rt').readlines(return_path=False)

    


    # Build a vocabulary object for these languages
    src_vocab = build_vocab_from_iterator(
        _yield_tokens(train_iterator_ja_src, src_tokenizer, True),
        min_freq=1,
        specials=list(special_symbols.keys()),
        special_first=True
    )

    tgt_vocab = None
    if tgt_lang == 'ja':
        tgt_vocab = build_vocab_from_iterator(
            _yield_tokens(train_iterator_ja_tgt, tgt_tokenizer, False),
            min_freq=1,
            specials=list(special_symbols.keys()),
            special_first=True
        )
    elif tgt_lang == 'cogs':
        tgt_vocab = build_vocab_from_iterator(
            _yield_tokens(train_iterator_cogs_tgt, tgt_tokenizer, False),
            min_freq=1,
            specials=list(special_symbols.keys()),
            special_first=True
        )
    else:
        raise NotImplementedError

    src_vocab.set_default_index(special_symbols["<unk>"])
    tgt_vocab.set_default_index(special_symbols["<unk>"])
    
    # Helper function to sequentially apply transformations
    def _seq_transform(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # Function to add BOS/EOS and create tensor for input sequence indices
    def _tensor_transform(token_ids):
        return torch.cat(
            (torch.tensor([special_symbols["<bos>"]]),
             torch.tensor(token_ids),
             torch.tensor([special_symbols["<eos>"]]))
        )
        

    src_lang_transform = _seq_transform(src_tokenizer, src_vocab, _tensor_transform)
    if opts.probe and opts.probe_mode == 'translation':
        train_iterator_ja_tgt = datapipes.iter.FileOpener([f"{opts.span_data_path}/translation/train_span_leace_constituency_ja.txt", f"{opts.span_data_path}/translation/val_span_leace_constituency_ja.txt", f"{opts.span_data_path}/translation/test_span_leace_constituency_ja.txt",f"{opts.span_data_path}/translation/gen_span_leace_constituency_ja.txt",], mode='rt').readlines(return_path=False)
        tgt_vocab_ = build_vocab_from_iterator(
            _yield_tokens(train_iterator_ja_tgt, tgt_tokenizer, False),
            min_freq=1,
            specials=list(special_symbols.keys()),
            special_first=True
        )
        tgt_lang_transform = _seq_transform(tgt_tokenizer, tgt_vocab_, _tensor_transform)
    else:
        tgt_lang_transform = _seq_transform(tgt_tokenizer, tgt_vocab, _tensor_transform)
        tgt_vocab_ = None

    # Now we want to convert the torchtext data pipeline to a dataloader. We
    # will need to collate batches
    def _collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_lang_transform(src_sample.rstrip("\n")))
            if skip or (not opts.probe and (eval or not opts.counterfactual)):
                tgt_batch.append(tgt_lang_transform(tgt_sample.rstrip("\n")))
            else:
                if 'leace' in opts.probe_mode or "random" in opts.probe_mode:
                    tgt_sample_list = tgt_sample.rstrip("\n").split('\t')
                    tgt_sample_list = [ast.literal_eval(x) for x in tgt_sample_list]
                    tgt_batch.append(
                        torch.tensor(tgt_sample_list)
                    )
                elif 'translation' in opts.probe_mode:
                    tgt_batch.append(tgt_lang_transform(tgt_sample.rstrip("\n")))
                else:
                    raise NotImplementedError

        src_batch = pad_sequence(src_batch, padding_value=special_symbols["<pad>"])
        if not opts.probe and (eval or not opts.counterfactual):
            tgt_batch = pad_sequence(tgt_batch, padding_value=special_symbols["<pad>"])
        else:
            tgt_batch = pad_sequence(tgt_batch, padding_value=0)
        return src_batch, tgt_batch
    
    def _collate_fn_id(batch):
        id_batch = []
        max_len = 0
        for _, gen_sample in batch:
            gen_len = src_lang_transform(gen_sample.rstrip("\n")).size(0)
            if gen_len > max_len:
                max_len = gen_len
        short_example = None
        for id_sample, _ in batch:
            id_tensor = src_lang_transform(id_sample.rstrip("\n"))
            if id_tensor.size(0) < max_len and short_example is None:
                short_example = id_tensor
            id_batch.append(id_tensor)
        new_id_batch = []
        for id_sample in id_batch:
            if id_sample.size(0) <= max_len:
                id_sample = torch.cat((id_sample, torch.tensor([special_symbols["<pad>"]]*(max_len-id_sample.size(0)))))
            else:
                id_sample = torch.cat((short_example, torch.tensor([special_symbols["<pad>"]]*(max_len-short_example.size(0)))))
            new_id_batch.append(id_sample)
        new_id_batch = pad_sequence(new_id_batch, padding_value=special_symbols["<pad>"])
        return new_id_batch

    if tgt_lang == 'ja':
        train_data = datapipes.iter.Zipper(train_iterator_ja_src, train_iterator_ja_tgt)
        valid_data = datapipes.iter.Zipper(dev_iterator_ja_src, dev_iterator_ja_tgt)
    elif tgt_lang == 'cogs':
        train_data = datapipes.iter.Zipper(train_iterator_ja_src, train_iterator_cogs_tgt)
        valid_data = datapipes.iter.Zipper(dev_iterator_ja_src, dev_iterator_cogs_tgt)
    else:
        raise NotImplementedError
    # Create the dataloader
    batch_size = opts.batch
    if opts.scrub:
        batch_size = 1
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=_collate_fn, shuffle=False)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=_collate_fn, shuffle=False)

    if skip and not eval:
        return train_dataloader, valid_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols
    
    if eval and opts.probe:
        hint_path = ""
        mode = opts.probe_mode
        if opts.probe_mode == "translation":
            hint_path = ""
            mode_ = "leace_constituency"
        else:
            hint_path = "counterfactual_hint"
            mode_ = opts.probe_mode
        tgt_test_path = f"{opts.span_data_path}/{mode}/{hint_path}/test_span_{mode_}_label.txt"
        tgt_gen_path = f"{opts.span_data_path}/{mode}/{hint_path}/gen_span_{mode_}_label.txt"
        src_test_path = f"{opts.span_data_path}/{mode}/{hint_path}/test_span_{mode_}_en.txt"
        src_gen_path = f"{opts.span_data_path}/{mode}/{hint_path}/gen_span_{mode_}_en.txt"
        
        src_iterator_test = datapipes.iter.FileOpener([src_test_path], mode='rt').readlines(return_path=False)
        tgt_iterator_test = datapipes.iter.FileOpener([tgt_test_path], mode='rt').readlines(return_path=False)
        src_iterator_gen = datapipes.iter.FileOpener([src_gen_path], mode='rt').readlines(return_path=False)
        tgt_iterator_gen = datapipes.iter.FileOpener([tgt_gen_path], mode='rt').readlines(return_path=False)
        test_data = datapipes.iter.Zipper(src_iterator_test, tgt_iterator_test)
        gen_data = datapipes.iter.Zipper(src_iterator_gen, tgt_iterator_gen)
        test_dataloader = DataLoader(test_data, batch_size=opts.batch, collate_fn=_collate_fn)
        gen_dataloader = DataLoader(gen_data, batch_size=opts.batch, collate_fn=_collate_fn)
        return test_dataloader, gen_dataloader, src_vocab, tgt_vocab, tgt_vocab_, src_lang_transform, tgt_lang_transform, special_symbols

    if eval:
        test_src_path = ""
        gen_src_path = ""
        test_tgt_path = ""
        gen_tgt_path = ""
        if tgt_lang == 'ja':
            test_src_path = f"{opts.data_path}/test_{src_lang}.txt"
            gen_src_path = f"{opts.data_path}/gen_test_{src_lang}.txt"
            test_tgt_path = f"{opts.data_path}/test_{tgt_lang}.txt"
            gen_tgt_path = f"{opts.data_path}/gen_test_{tgt_lang}.txt"
        elif tgt_lang == 'cogs':
            test_src_path = f"{opts.data_path}/test_{src_lang}.txt"
            gen_src_path = f"{opts.data_path}/gen_test_{src_lang}.txt"
            test_tgt_path = f"{opts.data_path}/test_lf.txt"
            gen_tgt_path = f"{opts.data_path}/gen_test_lf.txt"
        else:
            raise NotImplementedError

        test_iterator_src = datapipes.iter.FileOpener([test_src_path], mode='rt').readlines(return_path=False)
        test_iterator_tgt = datapipes.iter.FileOpener([test_tgt_path], mode='rt').readlines(return_path=False)
        gen_iterator_src = datapipes.iter.FileOpener([gen_src_path], mode='rt').readlines(return_path=False)
        gen_iterator_tgt = datapipes.iter.FileOpener([gen_tgt_path], mode='rt').readlines(return_path=False)                                                                                             

        test_data = datapipes.iter.Zipper(test_iterator_src, test_iterator_tgt)
        gen_data = datapipes.iter.Zipper(gen_iterator_src, gen_iterator_tgt)
        
        test_dataloader = DataLoader(test_data, batch_size=opts.batch, collate_fn=_collate_fn)
        gen_dataloader = DataLoader(gen_data, batch_size=opts.batch, collate_fn=_collate_fn)

        return test_dataloader, gen_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols
    
    if opts.prune:
        gen_src_train_path = f"{opts.data_path}/gen_train_{src_lang}.txt"
        gen_src_valid_path = f"{opts.data_path}/gen_valid_{src_lang}.txt"
        if tgt_lang == 'ja':
            gen_tgt_train_path = f"{opts.data_path}/gen_train_{tgt_lang}.txt"
            gen_tgt_valid_path = f"{opts.data_path}/gen_valid_{tgt_lang}.txt"
        elif tgt_lang == 'cogs':
            gen_tgt_train_path = f"{opts.data_path}/gen_train_lf.txt"
            gen_tgt_valid_path = f"{opts.data_path}/gen_valid_lf.txt"
        else:
            raise NotImplementedError
        gen_train_data = SGETDataset(gen_src_train_path, gen_tgt_train_path)
        gen_train_dataloader = DataLoader(gen_train_data, batch_size=opts.batch, collate_fn=_collate_fn)
        gen_valid_data = SGETDataset(gen_src_valid_path, gen_tgt_valid_path)
        gen_valid_dataloader = DataLoader(gen_valid_data, batch_size=opts.batch, collate_fn=_collate_fn)
        id_valid_data = SGETDataset(gen_src_valid_path, gen_src_valid_path)
        id_valid_dataloader = DataLoader(id_valid_data, batch_size=opts.batch, collate_fn=_collate_fn_id)
        return gen_train_dataloader, gen_valid_dataloader, id_valid_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols
    
    if opts.probe:
        mode = opts.probe_mode
        if opts.probe_mode == 'translation':
            hint_path = ""
            mode_ = "leace_constituency"
        else:
            hint_path = "counterfactual_hint"
            mode_ = opts.probe_mode
        src_train_path = f'{opts.span_data_path}/{mode}/{hint_path}/train_span_{mode_}_en.txt'
        tgt_train_path = f'{opts.span_data_path}/{mode}/{hint_path}/train_span_{mode_}_label.txt'
        src_valid_path = f'{opts.span_data_path}/{mode}/{hint_path}/val_span_{mode_}_en.txt'
        tgt_valid_path = f'{opts.span_data_path}/{mode}/{hint_path}/val_span_{mode_}_label.txt'
        
        src_iterator_train = datapipes.iter.FileOpener([src_train_path], mode='rt').readlines(return_path=False)
        tgt_iterator_train = datapipes.iter.FileOpener([tgt_train_path], mode='rt').readlines(return_path=False)
        src_iterator_valid = datapipes.iter.FileOpener([src_valid_path], mode='rt').readlines(return_path=False)
        tgt_iterator_valid = datapipes.iter.FileOpener([tgt_valid_path], mode='rt').readlines(return_path=False)
        train_data = datapipes.iter.Zipper(src_iterator_train, tgt_iterator_train)
        valid_data = datapipes.iter.Zipper(src_iterator_valid, tgt_iterator_valid)
        train_dataloader = DataLoader(train_data, batch_size=opts.batch, collate_fn=_collate_fn)
        valid_dataloader = DataLoader(valid_data, batch_size=opts.batch, collate_fn=_collate_fn)
        return train_dataloader, valid_dataloader, src_vocab, tgt_vocab, tgt_vocab_, src_lang_transform, tgt_lang_transform, special_symbols
    
    if opts.counterfactual:
        hint_path = ""
        if opts.probe_hint:
            hint_path = "counterfactual_hint"
        else:
            hint_path = "counterfactual_orig"
        mode = opts.probe_mode
        batch_size = opts.batch
        if opts.scrub:
            batch_size = 1
        src_train_path = f'{opts.span_data_path}/{mode}/{hint_path}/all_span_{opts.probe_mode}_en.txt'
        tgt_train_path = f'{opts.span_data_path}/{mode}/{hint_path}/all_span_{opts.probe_mode}_label.txt'
        src_valid_path = f'{opts.span_data_path}/{mode}/{hint_path}/val_span_{opts.probe_mode}_en.txt'
        tgt_valid_path = f'{opts.span_data_path}/{mode}/{hint_path}/val_span_{opts.probe_mode}_label.txt'
        src_iterator_train = datapipes.iter.FileOpener([src_train_path], mode='rt').readlines(return_path=False)
        tgt_iterator_train = datapipes.iter.FileOpener([tgt_train_path], mode='rt').readlines(return_path=False)
        src_iterator_valid = datapipes.iter.FileOpener([src_valid_path], mode='rt').readlines(return_path=False)
        tgt_iterator_valid = datapipes.iter.FileOpener([tgt_valid_path], mode='rt').readlines(return_path=False)
        train_data = datapipes.iter.Zipper(src_iterator_train, tgt_iterator_train)
        valid_data = datapipes.iter.Zipper(src_iterator_valid, tgt_iterator_valid)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=_collate_fn)
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=_collate_fn)
        return train_dataloader, valid_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols


    return train_dataloader, valid_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols

def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_square_subsequent_mask_batch(size, batch_num, opts, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0).repeat(opts.attn_heads*batch_num, 1, 1)
    return mask

# Create masks for input into model
def create_mask(src, tgt, pad_idx, device):

    # Get sequence length
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def gen_split(opts):
    '''
    split the generalization set for training masks for pruning
    '''
    gen_src_ja_path = f"{opts.data_path}/gen_entxt"
    gen_tgt_ja_path = f"{opts.data_path}/gen_ja.txt"
    gen_tgt_cogs_path = f"{opts.data_path}/gen_lf.txt"
    
    with open(gen_src_ja_path, 'r') as f:
        src_ja_data = f.readlines()
    with open(gen_tgt_ja_path, 'r') as f:
        tgt_ja_data = f.readlines()
    with open(gen_tgt_cogs_path, 'r') as f:
        tgt_cogs_data = f.readlines()
    gen_data = list(zip(src_ja_data, tgt_ja_data,  tgt_cogs_data))
    gen_train_ja_data = []
    gen_train_cogs_data = []
    gen_valid_ja_data = []
    gen_valid_cogs_data = []
    gen_test_ja_data = []
    gen_test_cogs_data = []

    for i, (src_ja, tgt_ja, tgt_cogs) in enumerate(gen_data):
        if i % 2 == 0:
            if i // 36000 < 1:
                gen_train_ja_data.append((src_ja, tgt_ja))
                gen_train_cogs_data.append((src_ja, tgt_cogs))
            else:
                continue
        elif i % 5 == 0:
            if i // 36000 < 1:
                gen_valid_ja_data.append((src_ja, tgt_ja))
                gen_valid_cogs_data.append((src_ja, tgt_cogs))
            else:
                continue
        else:
            if i // 36000 < 1:
                gen_test_ja_data.append((src_ja, tgt_ja))
                gen_test_cogs_data.append((src_ja, tgt_cogs))

    gen_src_ja_train_path = f"{opts.data_path}/gen_train_{opts.src}.txt"
    gen_tgt_ja_train_path = f"{opts.data_path}/gen_train_ja.txt"
    gen_tgt_cogs_train_path = f"{opts.data_path}/gen_train_lf.txt"
    gen_src_ja_valid_path = f"{opts.data_path}/gen_valid_{opts.src}.txt"
    gen_tgt_ja_valid_path = f"{opts.data_path}/gen_valid_ja.txt"
    gen_tgt_cogs_valid_path = f"{opts.data_path}/gen_valid_lf.txt"
    gen_src_ja_test_path = f"{opts.data_path}/gen_test_{opts.src}.txt"
    gen_tgt_ja_test_path = f"{opts.data_path}/gen_test_ja.txt"
    gen_tgt_cogs_test_path = f"{opts.data_path}/gen_test_lf.txt"

    with open(gen_src_ja_train_path, 'w') as f, open(gen_tgt_ja_train_path, 'w') as g:
        for src, tgt in gen_train_ja_data:
            f.write(src)
            g.write(tgt)
    with open(gen_tgt_cogs_train_path, 'w') as f:
        for _, tgt in gen_train_cogs_data:
            f.write(tgt)
    with open(gen_src_ja_valid_path, 'w') as f, open(gen_tgt_ja_valid_path, 'w') as g:
        for src, tgt in gen_valid_ja_data:
            f.write(src)
            g.write(tgt)
    with open(gen_tgt_cogs_valid_path, 'w') as f:
        for _, tgt in gen_valid_cogs_data:
            f.write(tgt)
    with open(gen_src_ja_test_path, 'w') as f, open(gen_tgt_ja_test_path, 'w') as g:
        for src, tgt in gen_test_ja_data:
            f.write(src)
            g.write(tgt)
    with open(gen_tgt_cogs_test_path, 'w') as f:
        for _, tgt in gen_test_cogs_data:
            f.write(tgt)