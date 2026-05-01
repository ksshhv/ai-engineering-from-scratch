import torch
import time

print("=== GPU Benchmark ===")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

size = 10000  # increase workload

# CPU benchmark
a = torch.randn(size, size)
b = torch.randn(size, size)

start = time.time()
c = a @ b
cpu_time = time.time() - start

# GPU benchmark
a_gpu = a.to(device).half()
b_gpu = b.to(device).half()

torch.cuda.synchronize()
start = time.time()

c_gpu = a_gpu @ b_gpu

torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.3f}s")
print(f"GPU time: {gpu_time:.3f}s")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
