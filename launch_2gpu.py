#!/usr/bin/env python3
"""
Launch distributed training - reads NUM_GPUS from train_distributed_llm.py
"""

import subprocess
import sys
import os

def get_num_gpus():
    """Read NUM_GPUS from the training script"""
    try:
        with open("train_distributed_llm.py", "r") as f:
            for line in f:
                if line.strip().startswith("NUM_GPUS = "):
                    return int(line.split("=")[1].strip())
    except:
        pass
    return 2  # default

def main():
    num_gpus = get_num_gpus()
    print(f"🚀 Launching distributed training on {num_gpus} GPUs...")
    
    # Check available GPUs
    import torch
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        print(f"🔍 Available GPUs: {available}")
        if num_gpus > available:
            print(f"⚠️  Requested {num_gpus} GPUs but only {available} available. Using {available}.")
            num_gpus = available
    else:
        print("❌ CUDA not available!")
        return
    
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
        print("❌ torchrun not found. Using torch.distributed.launch instead...")
        
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
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print(f"💡 Make sure you have {num_gpus} GPUs available and CUDA is properly installed")

if __name__ == "__main__":
    main()