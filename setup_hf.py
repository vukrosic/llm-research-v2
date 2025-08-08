#!/usr/bin/env python3
"""
Setup script for Hugging Face integration
"""

import subprocess
import sys
import os

def check_huggingface_cli():
    """Check if huggingface-cli is installed"""
    try:
        result = subprocess.run(['huggingface-cli', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ huggingface-cli is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ huggingface-cli not found")
    return False

def install_huggingface_hub():
    """Install huggingface_hub package"""
    print("📦 Installing huggingface_hub...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'], 
                      check=True)
        print("✅ huggingface_hub installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install huggingface_hub")
        return False

def check_login_status():
    """Check if user is logged in to Hugging Face"""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"✅ Logged in as: {username}")
            return username
    except FileNotFoundError:
        pass
    
    print("❌ Not logged in to Hugging Face")
    return None

def main():
    print("🤗 Hugging Face Setup for LLM Training")
    print("=" * 40)
    
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
        print("✅ huggingface_hub package is available")
    except ImportError:
        print("📦 Installing huggingface_hub package...")
        if not install_huggingface_hub():
            print("❌ Setup failed. Please install manually: pip install huggingface_hub")
            return
    
    # Check CLI installation
    if not check_huggingface_cli():
        print("\n📝 To install huggingface-cli:")
        print("   pip install huggingface_hub[cli]")
        print("   or")
        print("   pip install --upgrade huggingface_hub")
    
    # Check login status
    username = check_login_status()
    
    if not username:
        print("\n🔑 To login to Hugging Face:")
        print("   huggingface-cli login")
        print("   (You'll need a token from https://huggingface.co/settings/tokens)")
    
    print("\n📋 Next steps:")
    print("1. Update train_llm.py with your repository name:")
    print("   config.hf_repo_name = 'your-username/your-model-name'")
    print("2. Set config.push_to_hub = True")
    print("3. Run your training script!")
    
    if username:
        suggested_repo = f"{username}/minimal-llm-training"
        print(f"\n💡 Suggested repo name: {suggested_repo}")

if __name__ == "__main__":
    main()