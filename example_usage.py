#!/usr/bin/env python3
"""
Example usage of the training and upload scripts
"""

import os
import subprocess
import sys

def run_training():
    """Run the training script"""
    print("🚀 Starting model training...")
    print("This will train the model and save checkpoints every 5000 steps")
    
    # Run training
    result = subprocess.run([sys.executable, "train_llm.py"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        print("Checkpoints saved in ./checkpoints/ directory")
    else:
        print("❌ Training failed:")
        print(result.stderr)
        return False
    
    return True

def list_checkpoints():
    """List available checkpoints"""
    print("\n📋 Listing available checkpoints...")
    result = subprocess.run([sys.executable, "upload_to_hf.py", "--list"], 
                          capture_output=True, text=True)
    print(result.stdout)

def upload_checkpoint(repo_name, checkpoint_path=None, private=False):
    """Upload a checkpoint to Hugging Face"""
    print(f"\n🚀 Uploading checkpoint to Hugging Face: {repo_name}")
    
    # Check if HF_TOKEN is set
    if not os.getenv("HF_TOKEN"):
        print("❌ Please set your Hugging Face token:")
        print("export HF_TOKEN='your_token_here'")
        print("You can get a token from: https://huggingface.co/settings/tokens")
        return False
    
    # Build command
    cmd = [sys.executable, "upload_to_hf.py", "--repo-name", repo_name]
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    if private:
        cmd.append("--private")
    
    # Run upload
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Upload completed successfully!")
        print(result.stdout)
    else:
        print("❌ Upload failed:")
        print(result.stderr)
        return False
    
    return True

def main():
    print("🤖 LLM Training and Upload Example")
    print("=" * 40)
    
    # Example workflow
    print("\n1. First, let's train the model (this will take some time)...")
    input("Press Enter to start training, or Ctrl+C to skip...")
    
    try:
        if run_training():
            print("\n2. Let's see what checkpoints were created...")
            list_checkpoints()
            
            print("\n3. Now let's upload the latest checkpoint to Hugging Face...")
            repo_name = input("Enter your HF repo name (username/model-name): ").strip()
            
            if repo_name:
                private = input("Make repository private? (y/N): ").strip().lower() == 'y'
                upload_checkpoint(repo_name, private=private)
            else:
                print("No repo name provided, skipping upload")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
    
    print("\n📚 Manual Usage:")
    print("Training:")
    print("  python train_llm.py")
    print("\nList checkpoints:")
    print("  python upload_to_hf.py --list")
    print("\nUpload specific checkpoint:")
    print("  python upload_to_hf.py --repo-name username/model-name --checkpoint checkpoints/checkpoint_step_5000")
    print("\nUpload latest checkpoint:")
    print("  python upload_to_hf.py --repo-name username/model-name")

if __name__ == "__main__":
    main()