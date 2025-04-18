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

class QLinearTile2D_BSR(torch.nn.Module):
    def __init__(self, old: QLinearTile2D):
        super().__init__()
        self.out_features = old.out_features
        self.in_features = old.in_features
        self.tile_size = old.tile_size

        # 1a) Kwantyzacja dokładnie jak w forward()
        weight = old.weight.detach().cpu()
        R, C, TS = old.num_tiles_row, old.num_tiles_col, self.tile_size
        pad_rows = R*TS - self.out_features
        pad_cols = C*TS - self.in_features
        wp = F.pad(weight, (0, pad_cols, 0, pad_rows))
        tiles = wp.view(R, TS, C, TS).permute(0,2,1,3)
        e = old.e.detach().cpu().unsqueeze(-1).unsqueeze(-1)
        b = old.b.detach().cpu().unsqueeze(-1).unsqueeze(-1)
        qt = old.quantizer(tiles, e, b)
        qp = qt.permute(0,2,1,3).contiguous().view(R*TS, C*TS)
        q  = qp[:self.out_features, :self.in_features]

        # 1b) Konwersja do BSR
        self.weight_bsr = q.to_sparse_bsr(blocksize=(TS,TS)).to(old.weight.device)

    def forward(self, x):
        # obsłuż [B, T, C] i [B, C]
        if x.dim()==3:
            B, T, C = x.shape
            x2 = x.reshape(-1, C)
            y_t = torch.sparse.mm(self.weight_bsr, x2.t())
            return y_t.t().view(B, T, -1)
        else:
            y_t = torch.sparse.mm(self.weight_bsr, x.t())
            return y_t.t()

def inspect_and_save_quantization(layer: nn.Module, name="layer", save_dir="inspect_out"):
    import os, matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        raw_weight = layer.weight.detach().cpu()

        if isinstance(layer, QLinearPerChannel):
            e = layer.e.detach().cpu()
            b = layer.b.detach().cpu()
            quantized_weight = layer.quantizer(raw_weight, e, b)

            # Sparsity map: b == 0 per output channel
            sparsity_map = (F.relu(b) == 0).float().view(-1, 1) @ torch.ones(1, raw_weight.shape[1])

        elif isinstance(layer, QLinearTile2D):
            e = layer.e.detach().cpu()
            b = layer.b.detach().cpu()
            out_features = layer.out_features
            in_features = layer.in_features
            tile_size = layer.tile_size

            # Compute how much padding is applied
            num_tiles_row = math.ceil(out_features / tile_size)
            num_tiles_col = math.ceil(in_features / tile_size)
            padded_rows = num_tiles_row * tile_size
            padded_cols = num_tiles_col * tile_size

            # Pad weights manually for tiling
            pad_rows = padded_rows - out_features
            pad_cols = padded_cols - in_features
            weight_padded = F.pad(raw_weight, (0, pad_cols, 0, pad_rows))
            weight_tiles = weight_padded.view(num_tiles_row, tile_size,
                                              num_tiles_col, tile_size).permute(0, 2, 1, 3)

            e_exp = e.unsqueeze(-1).unsqueeze(-1)
            b_exp = b.unsqueeze(-1).unsqueeze(-1)
            quantized_tiles = layer.quantizer(weight_tiles, e_exp, b_exp)

            quantized_weight_padded = quantized_tiles.permute(0, 2, 1, 3).contiguous().view(
                padded_rows, padded_cols
            )
            quantized_weight = quantized_weight_padded[:out_features, :in_features]

            # Create tile-wise sparsity mask (b == 0)
            tile_mask = (F.relu(b) == 0).float()
            full_mask = tile_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)
            sparsity_map = full_mask[:out_features, :in_features]

        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        # Save raw data
        torch.save(raw_weight, f"{save_dir}/{name}_raw.pt")
        torch.save(quantized_weight, f"{save_dir}/{name}_quantized.pt")
        torch.save(b, f"{save_dir}/{name}_b.pt")

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].imshow(raw_weight, aspect='auto', cmap='bwr')
        axs[0].set_title("Raw Weight")

        axs[1].imshow(quantized_weight, aspect='auto', cmap='bwr')
        axs[1].set_title("Quantized Weight")

        axs[2].imshow(sparsity_map, cmap='Greys', aspect='auto')
        axs[2].set_title(f"Sparsity Map (b == 0)\nSparsity: {(sparsity_map.sum() / sparsity_map.numel()):.2%}")

        for ax in axs:
            ax.set_xlabel("In Features")
            ax.set_ylabel("Out Features")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name}_viz.png")
        plt.close()

        print(f"[{name}] Quantization inspection saved in: {save_dir}")


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

model.eval()
model.to(device)

# 2) Funkcja rekurencyjnie zastępująca moduły
def replace_qlinear_tiles(module):
    for name, child in list(module.named_children()):
        if isinstance(child, QLinearTile2D):
            setattr(module, name, QLinearTile2D_BSR(child))

model_sparse = GPT(cfg).eval().to(device)
model_sparse.load_state_dict(model.state_dict(), strict=False)
replace_qlinear_tiles(model_sparse)

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

for name, module in model.named_modules():
    if isinstance(module, (QLinearPerChannel, QLinearTile2D)):
        inspect_and_save_quantization(module, name=name.replace('.', '_'))

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

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
