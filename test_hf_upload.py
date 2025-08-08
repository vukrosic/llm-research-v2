#!/usr/bin/env python3
"""
Simple test script to verify Hugging Face upload functionality
"""
import os
from huggingface_hub import HfApi, create_repo, upload_file
from dotenv import load_dotenv
import tempfile
import json

def test_hf_connection():
    """Test basic HF connection and upload"""
    load_dotenv()
    
    token = os.getenv('HF_TOKEN')
    repo_name = os.getenv('HF_REPO_NAME', 'vukrosic/blueberry-1')
    
    if not token or token == 'hf_your_token_here':
        print("❌ Please set your HF_TOKEN in .env file")
        return False
    
    try:
        print(f"🔍 Testing connection to Hugging Face...")
        api = HfApi(token=token)
        
        # Test authentication
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        
        # Create test repo
        print(f"📦 Creating/verifying repo: {repo_name}")
        create_repo(repo_name, exist_ok=True, token=token)
        print(f"✅ Repo ready: https://huggingface.co/{repo_name}")
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test": "upload",
                "status": "working",
                "model_info": {
                    "architecture": "transformer",
                    "parameters": "test"
                }
            }
            json.dump(test_data, f, indent=2)
            temp_file = f.name
        
        # Upload test file
        print(f"📤 Uploading test file...")
        api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo="test_upload.json",
            repo_id=repo_name,
            commit_message="Test upload functionality",
            token=token
        )
        
        # Cleanup
        os.unlink(temp_file)
        
        print(f"🚀 Upload successful!")
        print(f"   View at: https://huggingface.co/{repo_name}/blob/main/test_upload.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("Make sure your token has write permissions")
        return False

if __name__ == "__main__":
    success = test_hf_connection()
    if success:
        print("\n✅ Hugging Face upload is working!")
        print("You can now run your training script with push_to_hub=True")
    else:
        print("\n❌ Fix the issues above before training")