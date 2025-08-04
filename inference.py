import scipy.sparse as sp
import torch
import torch.nn as nn
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from model import GPT, GPTConfig, QLinearTile2D
from sample import replace_qlinear_tiles, QLinearTile2D_BSR
import time

class CuPyLinear(torch.nn.Module):
    def __init__(self, sparse_weights):
        super(CuPyLinear, self).__init__()
        if not isinstance(sparse_weights, sp.csr_matrix):
            raise TypeError("sparse_weights must be a scipy.sparse.csr_matrix.")
         
        self.sparse_weights = sparse_weights
        self.weight_gpu = csr_matrix((cp.asarray(sparse_weights.data), cp.asarray(sparse_weights.indices), cp.asarray(sparse_weights.indptr)), shape=sparse_weights.shape)
    
    def forward(self, x):
        # (batch_size, seq_len, in_features)
        batch_size, seq_len, in_features = x.shape
        x_reshaped = x.view(-1, in_features).cpu().numpy()  # (batch_size * seq_len, in_features)
        x_cp = cp.asarray(x_reshaped.T)  # (in_features, batch_size * seq_len)
        result_cp = self.weight_gpu @ x_cp  # (out_features, batch_size * seq_len)
        result_np = cp.asnumpy(result_cp.T).reshape(batch_size, seq_len, -1)
        
        return torch.from_numpy(result_np).to(x.device)


def torch_linear_to_sparse(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D.")
    
    weight_array = tensor.detach().cpu().numpy()
    sparse_matrix = sp.csr_matrix(weight_array)
    
    return sparse_matrix

def load_model_from_checkpoint(ckpt_path):
    if not isinstance(ckpt_path, str):
        raise TypeError("Checkpoint path must be a string.")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    return model
    
def _replace_module_by_path(model: nn.Module, path: str, new_module: nn.Module):
    parts = path.split('.')
    submodule = model
    for p in parts[:-1]:
        submodule = getattr(submodule, p)
    setattr(submodule, parts[-1], new_module)

def convert_and_replace_linear(model: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, QLinearTile2D):
            sparse_weight = torch_linear_to_sparse(module.weight)
            new_module = CuPyLinear(sparse_weight)

            _replace_module_by_path(model, name, new_module)
            print(f"Replaced {name} with CuPyLinear")
    
    return model

def benchmark_model_inference(model, seq_len=128, batch_size=1, warmup=10, steps=50):
    model.eval()

    x = torch.randint(0, 65, (batch_size, seq_len), device='cuda')

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    def timed_inference(model):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        #torch.cuda.empty_cache()
        torch.cuda.synchronize()
        with torch.no_grad():
            start.record()
            for _ in range(steps):
                _ = model(x)
            end.record()
            end.synchronize()
        return start.elapsed_time(end) / steps  # ms

    time = timed_inference(model)
    return time


torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

ckpt_path = "./out-shakespeare-char/ckpt_tiled_quantized.pt"
ckpt_path_tiled = "./out-shakespeare-char/ckpt_tiled.pt"

model_dense = load_model_from_checkpoint(ckpt_path).cuda().eval()

model_sparse = load_model_from_checkpoint(ckpt_path).cuda().eval()
model_sparse_cupy = convert_and_replace_linear(model_sparse)

model_sparse_bsr = load_model_from_checkpoint(ckpt_path_tiled).cuda().eval()
replace_qlinear_tiles(model_sparse_bsr, sparsity_threshold=0.75)

print("\nRunning benchmark on dense vs sparse models")
time_dense = benchmark_model_inference(model_dense, seq_len=2*128, batch_size=4)
time_sparse_bsr = benchmark_model_inference(model_sparse_bsr, seq_len=2*128, batch_size=4)
time_sparse_cupy = benchmark_model_inference(model_sparse_cupy, seq_len=2*128, batch_size=4)

print(f"Dense model inference time: {time_dense:.2f} ms")
print(f"Sparse model (BSR) inference time: {time_sparse_bsr:.2f} ms")
print(f"Sparse model (CuPy) inference time: {time_sparse_cupy:.2f} ms")
print(f"Speedup BSR over Dense: {time_dense / time_sparse_bsr:.2f}x")
print(f"Speedup CuPy over Dense: {time_dense / time_sparse_cupy:.2f}x")
print(f"Speedup CuPy over BSR: {time_sparse_bsr / time_sparse_cupy:.2f}x")