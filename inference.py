import scipy.sparse as sp
import torch
import torch.nn as nn
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from model import GPT, GPTConfig, QLinearTile2D
import time

class CuPyLinear(torch.nn.Module):
    def __init__(self, sparse_weights):
        super(CuPyLinear, self).__init__()
        if not isinstance(sparse_weights, sp.csr_matrix):
            raise TypeError("sparse_weights must be a scipy.sparse.csr_matrix.")
        
        self.sparse_weights = sparse_weights
        self.weight_gpu = csr_matrix(sparse_weights.data, sparse_weights.indices, sparse_weights.indptr, shape=sparse_weights.shape)
    
    def forward(self, x):
        x_cp = cp.asarray(x.detach().cpu().numpy().T)
        result_cp = self.weight_gpu @ x_cp
        result = cp.asnumpy(result_cp.T)
        return torch.from_numpy(result).to(x.device)

def torch_linear_to_sparse(tensor, blocksize=(32, 32)):
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

def benchmark_model_inference(model_dense, model_sparse, seq_len=128, batch_size=1, warmup=10, steps=50):
    model_dense.eval()
    model_sparse.eval()

    x = torch.randint(0, model_dense.config.vocab_size, (batch_size, seq_len), device='cuda')

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_dense(x)
            _ = model_sparse(x)

    def timed_inference(model):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        with torch.no_grad():
            start.record()
            for _ in range(steps):
                _ = model(x)
            end.record()
            end.synchronize()
        return start.elapsed_time(end) / steps  # ms

    dense_time = timed_inference(model_dense)
    sparse_time = timed_inference(model_sparse)
    speedup = dense_time / sparse_time

    print(f"[Benchmark] Dense  : {dense_time:.2f} ms/iter")
    print(f"[Benchmark] Sparse : {sparse_time:.2f} ms/iter")
    print(f"[Benchmark] Speed-up: {speedup:.2f}×")

    return dense_time, sparse_time, speedup


torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

ckpt_path = "./quantized_models/ckpt_tiled_quantized.pt"
model_dense = load_model_from_checkpoint(ckpt_path).cuda().eval()
model_sparse = load_model_from_checkpoint(ckpt_path).cuda().eval()
model_sparse = convert_and_replace_linear(model_sparse)

print("\nRunning benchmark on dense vs sparse model")
benchmark_model_inference(model_dense, model_sparse, seq_len=128, batch_size=1)
