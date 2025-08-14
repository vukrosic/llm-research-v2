#!/usr/bin/env python3
"""
Improved launcher for 2x RTX 4090 training with balanced GPU utilization
"""

import os
import torch

# Enhanced memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:16"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow async operations for better utilization
os.environ["NCCL_DEBUG"] = "WARN"  # Reduce NCCL verbosity

# Set CPU affinity for better data loading performance
try:
    import psutil
    process = psutil.Process()
    cpu_count = psutil.cpu_count()
    # Use all available cores for data loading
    process.cpu_affinity(list(range(cpu_count)))
    print(f"🔧 Using all {cpu_count} CPU cores for data loading")
except ImportError:
    print("💡 Install psutil for optimal CPU utilization: pip install psutil")

# Clear GPU memory before starting to prevent fragmentation
if torch.cuda.is_available():
    print("🧹 Clearing GPU memory...")
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Print GPU info
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"🎯 GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

# Import and run
from train_distributed_llm import run_novita_4090_training

if __name__ == "__main__":
    print("🚀 Starting balanced 2x RTX 4090 training...")
    print("💡 Monitor GPU utilization with: python gpu_monitor.py")
    run_novita_4090_training()