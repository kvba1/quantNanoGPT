"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT, QLinearPerChannel, QLinearTile2D
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import copy
import os, matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

def bench_forward(model, name: str, x, ctx, warmup: int = 10, steps: int = 50):
    model.eval()
    torch.cuda.synchronize()
    # warmup
    with torch.no_grad(), ctx:
        for _ in range(warmup):
            _ = model(x)

    torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad(), ctx:
        for _ in range(steps):
            _ = model(x)
    torch.cuda.synchronize()
    t1 = time.time()

    ms = (t1 - t0) / steps * 1000
    if hasattr(model, "estimate_mfu"):
        tokens = x.numel() * steps            # tokens processed
        mfu = model.estimate_mfu(tokens, t1 - t0) * 100
        print(f"[{name}] {ms:7.2f} ms/iter | MFU {mfu:5.2f}%")
    else:
        print(f"[{name}] {ms:7.2f} ms/iter")
    return ms

def tensor_stats(t: torch.Tensor):
    flat = t.view(-1).float()
    numel = flat.numel()
    nz_cnt = torch.count_nonzero(flat).item()
    def near_zero(th): return int((flat.abs() < th).sum())
    return {
        "shape": list(t.shape),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "sparsity": 1.0 - nz_cnt / numel,
        "zero_count": int(numel - nz_cnt),
        "near_zero_1e3_count": near_zero(1e-3),
        "near_zero_1e4_count": near_zero(1e-4),
        "near_zero_1e5_count": near_zero(1e-5),
        "potential_sparsity_1e3": near_zero(1e-3) / numel,
        "potential_sparsity_1e4": near_zero(1e-4) / numel,
        "potential_sparsity_1e5": near_zero(1e-5) / numel,
    }
	
def _quantize_qlinear_per_channel(layer: QLinearPerChannel):
    with torch.no_grad():
        e = layer.e.detach()
        b = layer.b.detach()
        q_weight = layer.quantizer(layer.weight, e.unsqueeze(-1), b.unsqueeze(-1))
        layer.weight.copy_(q_weight)

def _quantize_qlinear_tile2d(layer: QLinearTile2D):
    with torch.no_grad():
        out_f, in_f, ts = layer.out_features, layer.in_features, layer.tile_size
        R, C_ = math.ceil(out_f / ts), math.ceil(in_f / ts)
        pad_r, pad_c = R * ts - out_f, C_ * ts - in_f
        wp = F.pad(layer.weight, (0, pad_c, 0, pad_r))
        tiles = wp.view(R, ts, C_, ts).permute(0, 2, 1, 3)
        e = layer.e.detach().unsqueeze(-1).unsqueeze(-1)
        b = layer.b.detach().unsqueeze(-1).unsqueeze(-1)
        qt = layer.quantizer(tiles, e, b)
        qp = qt.permute(0, 2, 1, 3).contiguous().view(R * ts, C_ * ts)
        q_weight = qp[:out_f, :in_f]
        layer.weight.copy_(q_weight)

def quantize_model_weights(model: torch.nn.Module):
    for name, mod in model.named_modules():
        if isinstance(mod, QLinearPerChannel):
            _quantize_qlinear_per_channel(mod)
        elif isinstance(mod, QLinearTile2D):
            _quantize_qlinear_tile2d(mod)
    return model

class QLinearTile2D_BSR(torch.nn.Module):
    def __init__(self, old: QLinearTile2D):
        super().__init__()
        self.out_features = old.out_features
        self.in_features = old.in_features
        self.tile_size = old.tile_size

        weight = old.weight.detach()
        R, C, TS = old.num_tiles_row, old.num_tiles_col, self.tile_size
        pad_rows = R*TS - self.out_features
        pad_cols = C*TS - self.in_features
        wp = F.pad(weight, (0, pad_cols, 0, pad_rows))
        tiles = wp.view(R, TS, C, TS).permute(0,2,1,3)
        e = old.e.detach().unsqueeze(-1).unsqueeze(-1)
        b = old.b.detach().unsqueeze(-1).unsqueeze(-1)
        qt = old.quantizer(tiles, e, b)
        qp = qt.permute(0,2,1,3).contiguous().view(R*TS, C*TS)
        q  = qp[:self.out_features, :self.in_features]

        self.weight_bsr = q.to_sparse_bsr(blocksize=(TS,TS)).to(old.weight.device)

    def forward(self, x):
        if x.dim()==3:
            B, T, C = x.shape
            x2 = x.reshape(-1, C)
            y_t = torch.sparse.mm(self.weight_bsr, x2.t())
            return y_t.t().view(B, T, -1)
        else:
            y_t = torch.sparse.mm(self.weight_bsr, x.t())
            return y_t.t()

def replace_qlinear_tiles(module, sparsity_threshold: float = 0.75):
    for name, child in list(module.named_children()):
        if isinstance(child, QLinearTile2D):
            with torch.no_grad():
                mask_zero = (F.relu(child.b) == 0)
                sparsity  = mask_zero.float().mean().item()          # sparsity % in a layer
            if sparsity > sparsity_threshold:
                setattr(module, name, QLinearTile2D_BSR(child))
                print(f"[sparse] {name}  sparsity={sparsity:.2%} to  BSR")
            else:
                print(f"[dense ] {name}  sparsity={sparsity:.2%}  (kept)")
        else:
            replace_qlinear_tiles(child, sparsity_threshold)


def inspect_and_save_quantization(layer: torch.nn.Module,
                                  name: str = "layer",
                                  save_dir: str = "inspect_out",
                                  n_heads: int = 6):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        raw_weight = layer.weight.detach().cpu()

        if isinstance(layer, QLinearPerChannel):
            e = layer.e.detach().cpu()
            b = layer.b.detach().cpu()
            quantized_weight = layer.quantizer(raw_weight, e, b)
            sparsity_map = (F.relu(b) == 0).float().view(-1, 1) @ \
                           torch.ones(1, raw_weight.shape[1])

        elif isinstance(layer, QLinearTile2D):
            e = layer.e.detach().cpu()
            b = layer.b.detach().cpu()
            out_f, in_f, ts = layer.out_features, layer.in_features, layer.tile_size

            R = math.ceil(out_f / ts)
            C_ = math.ceil(in_f  / ts)
            pad_r = R*ts - out_f
            pad_c = C_*ts - in_f
            wp = F.pad(raw_weight, (0, pad_c, 0, pad_r))
            tiles = wp.view(R, ts, C_, ts).permute(0,2,1,3)
            e_exp = e.unsqueeze(-1).unsqueeze(-1)
            b_exp = b.unsqueeze(-1).unsqueeze(-1)
            qt = layer.quantizer(tiles, e_exp, b_exp)
            qp = qt.permute(0,2,1,3).contiguous().view(R*ts, C_*ts)
            quantized_weight = qp[:out_f, :in_f]
            print(tensor_stats(quantized_weight))
            tile_mask = (F.relu(b) == 0).float()
            sparsity_map = tile_mask.repeat_interleave(ts, dim=0) \
                                .repeat_interleave(ts, dim=1)
            sparsity_map = sparsity_map[:out_f, :in_f]

        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        safe_name = name.replace('.', '_')
        torch.save(raw_weight,         f"{save_dir}/{safe_name}_raw.pt")
        torch.save(quantized_weight,   f"{save_dir}/{safe_name}_quantized.pt")
        if hasattr(layer, 'b'):
            torch.save(layer.b.detach().cpu(), f"{save_dir}/{safe_name}_b.pt")

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].imshow(raw_weight,       aspect='auto', cmap='bwr')
        axs[0].set_title("Raw Weight (3·C × C)")
        axs[1].imshow(quantized_weight, aspect='auto', cmap='bwr')
        axs[1].set_title("Quantized Weight")
        axs[2].imshow(sparsity_map,     aspect='auto', cmap='Greys')
        axs[2].set_title(f"Sparsity Map (b==0): {(sparsity_map.mean()*100):.2f}%")
        for ax in axs:
            ax.set_xlabel("In Features")
            ax.set_ylabel("Out Features")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{safe_name}_full_viz.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(quantized_weight.flatten().numpy(), bins=100)
        plt.title("Histogram of Quantized Weights")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name.replace('.', '_')}_hist.png")
        plt.close()

        if 'c_attn' not in name:
            print(f"[{name}] Quantization inspection saved in: {save_dir}")
            return
        print(f"[{name}] All c_attn quantization and head plots saved in: {save_dir}")


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt_tiled.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval().to(device)

for name, module in model.named_modules():
    if isinstance(module, (QLinearPerChannel, QLinearTile2D)):
        inspect_and_save_quantization(module, name=name.replace('.', '_'))

print("Quantising weights …", flush=True)
quantize_model_weights(model)
quant_ckpt_path = os.path.join(out_dir, "ckpt_tiled_quantized.pt")
torch.save({
    "model_args": checkpoint["model_args"],
    "model": model.state_dict(),
    "config": checkpoint.get("config", {})
}, quant_ckpt_path)
print(f"Saved quantized model to {quant_ckpt_path}")

sample_name, sample_layer = next(
    ((n, m) for n, m in model.named_modules() if isinstance(m, (QLinearPerChannel, QLinearTile2D))),
    (None, None),
)
if sample_layer:
    w = sample_layer.weight if hasattr(sample_layer, "weight") else sample_layer.weight_bsr.to_dense()
    s = tensor_stats(w)
    print(f"[quantised] Layer '{sample_name}' sparsity: {s['sparsity']*100:.2f}% | zero_count: {s['zero_count']} / {w.numel()}")

model_sparse = copy.deepcopy(model)
replace_qlinear_tiles(model_sparse)
print(model_sparse)
print(model_sparse.transformer.h[4].mlp.c_fc.weight_bsr)
print(model_sparse.transformer.h[4].mlp.c_proj.weight_bsr)

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

dense_ms  = bench_forward(model,        "Dense",  x, ctx)
sparse_ms = bench_forward(model_sparse, "Sparse", x, ctx)
print(f"Speed-up: {dense_ms/sparse_ms:5.2f}×")

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model_sparse.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
