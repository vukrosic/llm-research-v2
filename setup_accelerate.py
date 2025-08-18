#!/usr/bin/env python3
"""
Setup script for Hugging Face Accelerate on RTX 4090s
This helps configure Accelerate for optimal GPU balance
"""

import subprocess
import sys
import os

def check_accelerate():
    """Check if Accelerate is installed"""
    try:
        import accelerate
        print(f"✅ Accelerate {accelerate.__version__} is installed")
        return True
    except ImportError:
        print("❌ Accelerate not installed")
        return False

def install_accelerate():
    """Install Accelerate if not present"""
    print("�� Installing Accelerate...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "accelerate"], check=True)
        print("✅ Accelerate installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Accelerate")
        return False

def configure_accelerate():
    """Configure Accelerate for RTX 4090s"""
    print("🔧 Configuring Accelerate for RTX 4090s...")
    
    # Create a configuration file
    config_content = """compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '0,1'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    config_dir = os.path.expanduser("~/.cache/huggingface/accelerate")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "default_config.yaml")
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Accelerate configured for 2x RTX 4090s")
    print(f"�� Config saved to: {config_file}")

def run_training():
    """Run the training with proper configuration"""
    print("�� Launching balanced training...")
    
    # Set environment variables for better GPU balance
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Launch training
    cmd = [
        "accelerate", "launch",
        "--config_file", os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml"),
        "train_distributed_llm_accelerate_balanced.py"
    ]
    
    print(f"�� Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, env=env, check=True)
        print("✅ Training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("💡 Check the error messages above for troubleshooting")

def main():
    """Main setup function"""
    print("=" * 80)
    print("🚀 RTX 4090 ACCELERATE SETUP")
    print("=" * 80)
    
    # Check and install Accelerate
    if not check_accelerate():
        if not install_accelerate():
            print("❌ Cannot proceed without Accelerate")
            return
    
    # Configure Accelerate
    configure_accelerate()
    
    # Ask user if they want to run training
    print("\n" + "=" * 80)
    response = input("🚀 Ready to launch training? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_training()
    else:
        print("💡 To run training later, use:")
        print("   accelerate launch train_distributed_llm_accelerate_balanced.py")
        print("   or")
        print("   python setup_accelerate.py")

if __name__ == "__main__":
    main() 