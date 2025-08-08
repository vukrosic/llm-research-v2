#!/usr/bin/env python3
"""
Simple setup script for the LLM training project
"""
import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def setup_env_file():
    """Create .env file from template if it doesn't exist"""
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        print("📝 Creating .env file from template...")
        with open('.env.example', 'r') as src, open('.env', 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file - edit it to configure Hugging Face upload")
    elif os.path.exists('.env'):
        print("✅ .env file already exists")
    else:
        print("⚠️ No .env.example found")

def main():
    print("🫐 LLM Training Project Setup")
    print("=" * 30)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Setup environment file
    setup_env_file()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Try the pre-trained model: python inference.py")
    print("2. Train your own model: python train_llm.py")
    print("3. Read the full README.md for detailed instructions")
    
    if os.path.exists('.env'):
        print("4. Edit .env file to configure Hugging Face upload")

if __name__ == "__main__":
    main()