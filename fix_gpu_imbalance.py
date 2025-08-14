#!/usr/bin/env python3
"""
Quick fixes for GPU utilization imbalance in distributed training
"""

import torch
import torch.distributed as dist
import os
import subprocess
import sys

def check_gpu_setup():
    """Check GPU setup and identify potential issues"""
    print("🔍 Diagnosing GPU setup...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPUs")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    return True

def fix_memory_fragmentation():
    """Clear GPU memory and set optimal memory settings"""
    print("🧹 Clearing GPU memory fragmentation...")
    
    # Clear cache on all GPUs
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:16"
    print("✅ Memory optimization settings applied")

def check_process_affinity():
    """Check CPU affinity and suggest improvements"""
    print("🔍 Checking process affinity...")
    
    try:
        import psutil
        current_process = psutil.Process()
        cpu_affinity = current_process.cpu_affinity()
        cpu_count = psutil.cpu_count()
        
        print(f"  CPU cores available: {cpu_count}")
        print(f"  Process affinity: {len(cpu_affinity)} cores")
        
        if len(cpu_affinity) < cpu_count:
            print("⚠️  Process is not using all available CPU cores")
            print("   This can cause data loading bottlenecks")
    except ImportError:
        print("  (Install psutil for detailed CPU analysis)")

def suggest_batch_size():
    """Suggest optimal batch size based on GPU memory"""
    print("📊 Analyzing optimal batch size...")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        
        # Conservative batch size estimation
        if memory_gb >= 20:  # RTX 4090 class
            suggested_batch = 8
        elif memory_gb >= 16:  # RTX 4080 class
            suggested_batch = 6
        elif memory_gb >= 12:  # RTX 4070 class
            suggested_batch = 4
        else:
            suggested_batch = 2
        
        print(f"  GPU {i} ({memory_gb:.1f}GB): Suggested batch size = {suggested_batch}")

def create_balanced_launcher():
    """Create a balanced training launcher script"""
    launcher_script = """#!/usr/bin/env python3
import os
import torch

# Optimal settings for balanced GPU utilization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:16"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow async operations
os.environ["NCCL_DEBUG"] = "WARN"  # Reduce NCCL verbosity

# Set CPU affinity for better data loading
try:
    import psutil
    process = psutil.Process()
    cpu_count = psutil.cpu_count()
    # Use all available cores
    process.cpu_affinity(list(range(cpu_count)))
except ImportError:
    pass

# Clear GPU memory before starting
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# Import and run training
from train_distributed_llm import run_novita_4090_training

if __name__ == "__main__":
    print("🚀 Starting balanced GPU training...")
    run_novita_4090_training()
"""
    
    with open("run_balanced.py", "w") as f:
        f.write(launcher_script)
    
    print("✅ Created run_balanced.py with optimized settings")

def main():
    print("🔧 GPU Imbalance Diagnostic and Fix Tool")
    print("=" * 50)
    
    if not check_gpu_setup():
        return
    
    fix_memory_fragmentation()
    check_process_affinity()
    suggest_batch_size()
    create_balanced_launcher()
    
    print("\n🎯 Recommendations to fix GPU imbalance:")
    print("1. Use the created 'run_balanced.py' script instead of 'run_4090.py'")
    print("2. Monitor training with: python gpu_monitor.py")
    print("3. If imbalance persists, try reducing batch size further")
    print("4. Consider using PyTorch's native DDP instead of custom distributed optimizers")
    print("5. Ensure your data loading is not the bottleneck (use more workers)")
    
    print("\n🚀 Quick test command:")
    print("python run_balanced.py")

if __name__ == "__main__":
    main()