#!/usr/bin/env python3
"""
Quick setup script to configure your Hugging Face settings
"""

def main():
    print("🤗 Hugging Face Configuration Setup")
    print("=" * 40)
    
    # Get user input
    repo_name = input("Enter your HF repo name (e.g., username/model-name): ").strip()
    if not repo_name:
        repo_name = "vukrosic/blueberry-1"  # default
        print(f"Using default: {repo_name}")
    
    token = input("Enter your HF token (starts with hf_): ").strip()
    if not token:
        print("❌ Token is required! Get one from https://huggingface.co/settings/tokens")
        return
    
    # Read the training file
    with open('train_llm.py', 'r') as f:
        content = f.read()
    
    # Replace the placeholders
    content = content.replace('config.hf_repo_name = "vukrosic/blueberry-1"', 
                             f'config.hf_repo_name = "{repo_name}"')
    content = content.replace('config.hf_token = "hf_your_token_here"', 
                             f'config.hf_token = "{token}"')
    
    # Write back
    with open('train_llm.py', 'w') as f:
        f.write(content)
    
    print(f"✅ Configuration updated!")
    print(f"   Repo: {repo_name}")
    print(f"   Token: {token[:10]}...")
    print(f"\n🚀 Ready to run: python train_llm.py")

if __name__ == "__main__":
    main()