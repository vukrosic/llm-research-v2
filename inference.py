import torch
import torch.nn.functional as F
from train_llm import MinimalLLM, ModelConfig
from transformers import AutoTokenizer
from huggingface_hub import HfApi, snapshot_download
import json
import os
import glob
import re

def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    # Load config
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object
    config = ModelConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # Load model
    model = MinimalLLM(config)
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return model, config

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50):
    """Generate text from prompt"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # Decode and return
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def get_hf_checkpoints(repo_name="vukrosic/blueberry-1"):
    """Get list of checkpoints available on Hugging Face"""
    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_name)
        
        # Find checkpoint directories
        hf_checkpoints = set()
        for file_path in repo_files:
            if file_path.startswith("checkpoint-") and "/" in file_path:
                checkpoint_name = file_path.split("/")[0]
                if re.match(r"checkpoint-\d+", checkpoint_name):
                    hf_checkpoints.add(checkpoint_name)
        
        return sorted(list(hf_checkpoints), key=lambda x: int(x.split('-')[-1]))
    except Exception as e:
        print(f"⚠️ Could not fetch HF checkpoints: {e}")
        return []

def download_checkpoint(repo_name, checkpoint_name, local_dir="checkpoints"):
    """Download a specific checkpoint from HF"""
    try:
        print(f"📥 Downloading {checkpoint_name}...")
        snapshot_download(
            repo_id=repo_name,
            allow_patterns=f"{checkpoint_name}/*",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded {checkpoint_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {checkpoint_name}: {e}")
        return False

def main():
    print("🤖 LLM Inference Tool")
    
    # Check HF repo for additional checkpoints
    print("🔍 Checking Hugging Face repo for checkpoints...")
    hf_checkpoints = get_hf_checkpoints()
    
    # Find local checkpoints
    local_checkpoints = glob.glob("checkpoints/checkpoint-*")
    local_checkpoint_names = [os.path.basename(cp) for cp in local_checkpoints]
    
    # Find missing checkpoints
    missing_checkpoints = [cp for cp in hf_checkpoints if cp not in local_checkpoint_names]
    
    if missing_checkpoints:
        print(f"\n📦 Found {len(missing_checkpoints)} additional checkpoints on HF:")
        for i, checkpoint in enumerate(missing_checkpoints):
            step = checkpoint.split('-')[-1]
            print(f"  {i+1}. Step {step} ({checkpoint})")
        
        download_choice = input(f"\nDownload checkpoints? (y/n/select): ").strip().lower()
        
        if download_choice == 'y':
            # Download all missing
            for checkpoint in missing_checkpoints:
                download_checkpoint("vukrosic/blueberry-1", checkpoint)
        elif download_choice == 'select':
            # Let user select which to download
            print("Select checkpoints to download (comma-separated numbers, e.g., 1,3,5):")
            try:
                selections = input("Numbers: ").strip().split(',')
                for sel in selections:
                    idx = int(sel.strip()) - 1
                    if 0 <= idx < len(missing_checkpoints):
                        download_checkpoint("vukrosic/blueberry-1", missing_checkpoints[idx])
            except ValueError:
                print("Invalid selection, skipping downloads.")
    
    # Refresh local checkpoints after potential downloads
    checkpoints = glob.glob("checkpoints/checkpoint-*")
    if not checkpoints:
        print("❌ No checkpoints found in 'checkpoints/' directory")
        return
    
    # Sort checkpoints by step number
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    
    print(f"\n📁 Available checkpoints:")
    for i, checkpoint in enumerate(checkpoints):
        step = checkpoint.split('-')[-1]
        print(f"  {i+1}. Step {step} ({checkpoint})")
    
    # Let user choose
    while True:
        try:
            choice = int(input(f"\nChoose checkpoint (1-{len(checkpoints)}): ")) - 1
            if 0 <= choice < len(checkpoints):
                selected_checkpoint = checkpoints[choice]
                break
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a number.")
    
    print(f"Loading checkpoint: {selected_checkpoint}")
    
    # Load model and tokenizer
    model, config = load_model_from_checkpoint(selected_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✅ Model loaded on {device}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Interactive generation
    print("\n🎯 Interactive Text Generation (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            continue
            
        print("\nGenerating...")
        generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8)
        print(f"\n📝 Generated text:\n{generated}")
        print("-" * 50)

if __name__ == "__main__":
    main()