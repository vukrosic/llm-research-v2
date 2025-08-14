#!/usr/bin/env python3
"""
Launcher script for distributed training of the LLM.
This script handles the proper setup of environment variables for PyTorch distributed training.
"""

import subprocess
import sys
import os
import torch

def launch_single_gpu():
    """Launch training on a single GPU"""
    print("🚀 Launching single GPU training...")
    cmd = [sys.executable, "train_distributed_llm.py"]
    subprocess.run(cmd)

def launch_multi_gpu(num_gpus: int):
    """Launch distributed training on multiple GPUs"""
    print(f"🚀 Launching distributed training on {num_gpus} GPUs...")
    
    # Use torchrun for proper distributed setup
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
        print("❌ torchrun not found. Installing torch with distributed support...")
        print("💡 Alternative: Use python -m torch.distributed.launch")
        
        # Fallback to torch.distributed.launch
        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            f"--nproc_per_node={num_gpus}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=12355",
            "train_distributed_llm.py"
        ]
        subprocess.run(cmd, check=True)

def main():
    # Check available GPUs
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Running on CPU...")
        launch_single_gpu()
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Found {num_gpus} GPU(s)")
    
    if num_gpus == 1:
        print("📱 Single GPU detected")
        choice = input("Run on single GPU? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '']:
            launch_single_gpu()
        else:
            print("Exiting...")
    else:
        print(f"🖥️ Multiple GPUs detected: {num_gpus}")
        print("Options:")
        print("1. Single GPU training")
        print(f"2. Multi-GPU training ({num_gpus} GPUs)")
        
        choice = input("Choose option (1/2): ").strip()
        
        if choice == "1":
            launch_single_gpu()
        elif choice == "2":
            launch_multi_gpu(num_gpus)
        else:
            print("Invalid choice. Defaulting to single GPU...")
            launch_single_gpu()

if __name__ == "__main__":
    main()