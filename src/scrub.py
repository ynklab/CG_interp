from concept_erasure import LeaceFitter, LeaceEraser
from data import get_data, create_mask
from model import Translator, Translator_mask
from main import beam_decode
import torch

from contextlib import contextmanager
from functools import partial
from typing import Callable

from torch import Tensor, nn
from typing import Any, Type, TypeVar, cast

from torch import nn
import numpy as np
import random
import argparse
import logging
import os
from tqdm import tqdm


T = TypeVar("T")

# https://github.com/EleutherAI/concept-erasure/blob/main/concept_erasure/
def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def is_norm_layer(module: nn.Module) -> bool:
    """Return `True` if the module is a normalization layer."""
    cls_name = type(module).__name__
    return cls_name.endswith("LayerNorm") or cls_name.endswith("RMSNorm")


def mangle_module_path(name: str) -> str:
    """Mangle a module path to make it a valid key in a `nn.ModuleDict`."""
    # This is a weird edge case we probably don't need to support
    assert "-" not in name, "Module path cannot contain `-`"
    return name.replace(".", "-")

def evaluate(model, test_dl, tgt_vocab, special_symbols, opts, device):
    test_preds = []
    for src, tgt in test_dl:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = (torch.zeros(opts.attn_heads*src.shape[1], src.shape[0], src.shape[0])).type(torch.bool)
        src_padding_mask = (src == special_symbols["<pad>"]).transpose(0, 1)
        
        with torch.no_grad():
            tgt_tokens = beam_decode(
                model, src, src_mask, src_padding_mask, tgt_vocab, max_len=256, start_symbol=special_symbols["<bos>"], end_symbol=special_symbols["<eos>"], pad_symbol=special_symbols["<pad>"], opts=opts, device=device
            )
        output_as_list = tgt_tokens
        output_list_words = [tgt_vocab.lookup_tokens(i) for i in output_as_list]
        translations = [" ".join(words).replace('<bos>', '').replace('<eos>', '').replace('<pad>', "") for words in output_list_words]
        test_preds.extend(translations)
    return test_preds

class ConceptScrubber:
    """Wrapper for a dictionary mapping module paths to `LeaceEraser` objects."""

    def __init__(self, pre_hook: bool = False):
        super().__init__()

        self.erasers: dict[str, LeaceEraser] = {}
        self.pre_hook = pre_hook

    @contextmanager
    def scrub(self, model):
        """Add hooks to the model which apply the erasers during a forward pass."""

        def scrub_hook(key: str, x: Tensor):
            eraser = assert_type(LeaceEraser, self.erasers[key])
            return eraser(x).type_as(x)

        with self.apply_hook(model, scrub_hook):
            yield self

    @contextmanager
    def apply_hook(
        self,
        model: nn.Module,
        hook_fn: Callable[[str, Tensor], Tensor | None],
    ):
        """Apply a `hook_fn` to each submodule in `model` that we're scrubbing."""

        def post_wrapper(_, __, output, name: str) -> Tensor | None:
            key = mangle_module_path(name)
            return hook_fn(key, output)

        def pre_wrapper(_, inputs, name: str) -> tuple[Tensor | None, ...]:
            x, *extras = inputs
            key = mangle_module_path(name)
            return hook_fn(key, x), *extras

        handles = [
            (
                mod.register_forward_pre_hook(partial(pre_wrapper, name=name))
                if self.pre_hook
                else mod.register_forward_hook(partial(post_wrapper, name=name))
            )
            for name, mod in model.named_modules()
            if is_norm_layer(mod) and mangle_module_path(name) in self.erasers
        ]
        assert len(handles) == len(self.erasers), "Not all erasers can be applied"

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()

@torch.no_grad()
def scrub(model, train_dl, mt_train_dl, opts, device, special_symbols):

    xs = []
    zs = []
    masks = []
    scrubber = ConceptScrubber()
    label_dim = 20 if 'leace_constituency' in opts.probe_mode or "random" in opts.probe_mode else 12
    enc_layers, dec_layers = model.get_layers()
    for batch, mt_batch in zip(train_dl, mt_train_dl):
        src, tgt = batch
        _, tgt_ja = mt_batch
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_ja = tgt_ja.to(device)
        tgt_input = tgt_ja[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], device)
        src_emb = model.src_embedding(src)
        pos_enc = model.pos_enc(src_emb, model.embed_size, model.embed_scale)
        xs.append(pos_enc)
        zs.append(tgt)
        masks.append((src_mask, tgt_mask, src_padding_mask, tgt_padding_mask))
    
    for i, enc_layer in enumerate(tqdm(enc_layers)):
        attn_fitter = LeaceFitter(
            model.embed_size, label_dim, affine=True, device=device, method='leace'
        )
        for x, z, mask in zip(xs, zs, masks):
            src_mask, _, src_padding_mask, _ = mask
            x_attn = enc_layer._sa_block(x, src_mask, src_padding_mask)
            x = enc_layer.norm1(x + x_attn)
            attn_fitter.update(x, z)

        attn_eraser = attn_fitter.eraser
        scrubber.erasers[f'transformer-encoder-layers-{i}-norm1'] = attn_eraser
        del attn_fitter

        for j, (x, mask) in enumerate(zip(xs, masks)):
            src_mask, _, src_padding_mask, _ = mask
            x_attn = enc_layer._sa_block(x, src_mask, src_padding_mask)
            h = enc_layer.norm1(x + x_attn)
            h = attn_eraser(h).type_as(h)
            xs[j] = h
        
        mlp_fitter = LeaceFitter(
            model.embed_size, label_dim, affine=True, device=device, method='leace'
        )
        for x, z in zip(xs, zs):
            x_mlp = enc_layer._ff_block(x)
            x = enc_layer.norm2(x + x_mlp)
            mlp_fitter.update(x, z)

        mlp_eraser = mlp_fitter.eraser
        scrubber.erasers[f'transformer-encoder-layers-{i}-norm2'] = mlp_eraser
        del mlp_fitter

        for j, x in enumerate(xs):
            x_mlp = enc_layer._ff_block(x)
            h = enc_layer.norm2(x + x_mlp)
            h = mlp_eraser(h).type_as(h)
            xs[j] = h
    return scrubber

def main(opts, device):
    train_dl, val_dl, _, _,  _, _, special_symbols = get_data(opts, eval=False)
    mt_train_dl, mt_val_dl, src_vocab, tgt_vocab, _, _, _ = get_data(opts, eval=False, skip=True)
    test_dl, gen_dl, _, _, _, _, _ = get_data(opts, eval=True)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    if args.is_masked:
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
        ).to(device)
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
            embed_scale=opts.embed_scale,
        ).to(device)
        model.load_state_dict(torch.load(opts.model_path))

    scrubber = scrub(model, train_dl, mt_train_dl, opts, device, special_symbols)
    original_model_dict = model.state_dict()
    with scrubber.scrub(model):
        logger.info("Save scrubbed model to %s", opts.logging_dir + "scrubbed_model.pt")
        torch.save(model.state_dict(), opts.logging_dir + "scrubbed_model.pt")
        test_results = evaluate(model, test_dl, tgt_vocab, special_symbols, opts, device)
        gen_results = evaluate(model, gen_dl, tgt_vocab, special_symbols, opts, device)

    with open(f"{opts.result_path}/pred_test.txt", 'w') as f:
        f.write("\n".join(test_results))
    with open(f"{opts.result_path}/pred_gen.txt", 'w') as f:
        f.write("\n".join(gen_results))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--src', type=str, help='source language')
    parser.add_argument('--tgt', type=str, help='target language')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument("--data_train_path", type=str, default="../../../data/restrict/train",)
    parser.add_argument('--span_data_path', type=str, help='path to span data')
    parser.add_argument("--prune", action="store_true",
                        help="Set true to run pruning")
    parser.add_argument("--ablate", action="store_true",
                        help="Set true to reverse mask", default=False)
    parser.add_argument('--is_masked', action='store_true', help='whether the model is masked')
    parser.add_argument('--probe', action='store_true', help='whether to probe')
    parser.add_argument('--probe_mode', type=str, help='mode of span', default='iobj')
    parser.add_argument('--probe_hint', action='store_true', help='whether to include hints')
    parser.add_argument('--counterfactual', action='store_true', help='whether to run counterfactual')
    parser.add_argument('--scrub', action='store_true', help='whether to scrub')
    parser.add_argument("--pos_enc", type=str, default="default",)
    parser.add_argument("--hint", type=str, default="default",
                        help="Set true to run hint")
    parser.add_argument("--attn_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument('--enc_layers', type=int, help='number of encoder layers')
    parser.add_argument('--dec_layers', type=int, help='number of decoder layers')
    parser.add_argument('--embed_size', type=int, default=512, help='embedding dimension')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--embed_init', type=str, help='embedding initialization')
    parser.add_argument('--embed_scale', type=str, help='embedding scale')
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--logging_dir', type=str, help='logging directory')
    parser.add_argument("--result_path", type=str, default="../../../results",
                        help="Path to the results")
    parser.add_argument("--random", action="store_true",)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=args.logging_dir + "log.txt", level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    main(args, device)
