Check how much GPU memory you have and estimate the largest model you can fit (rule of thumb: 2 bytes per parameter for fp16) 

This is the my GPU Settings:

CUDA available: True 
CUDA version: 12.4 
GPU: NVIDIA GeForce RTX 3070 Ti Laptop 
GPU Memory: 8.2 GB

🔹 What fits comfortably?  
✅ 2B–3B models (FP16) → smooth
✅ 7B models → with 4-bit quantization  
⚠️ 13B → only with aggressive quantization + offloading