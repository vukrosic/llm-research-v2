#!/usr/bin/env python3
"""
Simple training runner that automatically handles single vs multi-GPU setup
"""

import os
import torch
import subprocess
import sys

def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please install CUDA-enabled PyTorch")
        print("💡 Visit: https://pytorch.org/get-started/locally/")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Found {num_gpus} GPU(s): {torch.cuda.get_device_name()}")
    
    if num_gpus == 1:
        print("🚀 Starting single GPU training...")
        # Set environment for single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Import and run directly
        try:
            from train_distributed_llm import main as train_main
            train_main()
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Try running: python train_llm.py (non-distributed version)")
    
    else:
        print(f"🚀 Starting multi-GPU training on {num_gpus} GPUs...")
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--nnodes=1", 
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=12355",
            "train_distributed_llm.py"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print("❌ torchrun not found. Falling back to single GPU...")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            from train_distributed_llm import main as train_main
            train_main()

if __name__ == "__main__":
    main()