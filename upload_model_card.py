#!/usr/bin/env python3
"""
Upload model card and visualizations to Hugging Face Hub
"""
import os
from huggingface_hub import HfApi, upload_file
from dotenv import load_dotenv

def upload_model_card():
    """Upload README and images to HF Hub"""
    load_dotenv()
    
    token = os.getenv('HF_TOKEN')
    repo_name = os.getenv('HF_REPO_NAME', 'vukrosic/blueberry-1')
    
    if not token:
        print("❌ Please set HF_TOKEN in .env file")
        return
    
    api = HfApi(token=token)
    
    files_to_upload = [
        ('README.md', 'README.md'),
        ('training_curves.png', 'training_curves.png'),
        ('model_architecture.png', 'model_architecture.png'),
        ('performance_comparison.png', 'performance_comparison.png')
    ]
    
    print(f"📤 Uploading model card to {repo_name}")
    
    for local_file, repo_file in files_to_upload:
        if os.path.exists(local_file):
            try:
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=repo_file,
                    repo_id=repo_name,
                    commit_message=f"Add {repo_file}",
                    token=token
                )
                print(f"✅ Uploaded {local_file}")
            except Exception as e:
                print(f"❌ Failed to upload {local_file}: {e}")
        else:
            print(f"⚠️ File not found: {local_file}")
    
    print(f"\n🎉 Model card uploaded!")
    print(f"View at: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    upload_model_card()