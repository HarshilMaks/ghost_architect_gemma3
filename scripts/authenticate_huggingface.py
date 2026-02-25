"""
Helper script to authenticate with HuggingFace for gated models.
Run this in Colab before training with Phase 2 (train_vision.py)
"""

import os
import sys
from pathlib import Path

def authenticate():
    """Authenticate with HuggingFace using token from .env or interactive prompt."""
    
    try:
        from huggingface_hub import login, whoami
        from dotenv import load_dotenv
    except ImportError:
        print("❌ Missing required packages")
        print("Run: pip install huggingface-hub python-dotenv")
        sys.exit(1)
    
    print("\n🔐 HuggingFace Authentication")
    print("=" * 50)
    
    # Try loading from .env first
    load_dotenv()
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    
    if hf_token:
        print("✓ Found HUGGINGFACE_TOKEN in .env")
        try:
            login(token=hf_token, add_to_git_credential=False)
            user = whoami()
            print(f"✅ Authenticated as: {user['name']}")
            return True
        except Exception as e:
            print(f"❌ Token from .env invalid: {e}")
            print("Please create a new token at: https://huggingface.co/settings/tokens")
            return False
    else:
        print("⚠️  HUGGINGFACE_TOKEN not found in .env")
        print("\nOptions:")
        print("  1. Add token to .env file (recommended for Colab)")
        print("  2. Use interactive login below")
        
        # In Colab, offer interactive login
        try:
            from huggingface_hub import notebook_login
            print("\n📝 Starting interactive login...")
            notebook_login()
            return True
        except Exception:
            print("\n❌ Automatic authentication failed")
            print("Manual steps:")
            print("  1. Get token: https://huggingface.co/settings/tokens")
            print("  2. Add to .env: HUGGINGFACE_TOKEN=hf_...")
            print("  3. Rerun this script")
            return False

def verify_model_access():
    """Verify you have access to Gemini-3-12B model."""
    try:
        from huggingface_hub import model_info
        
        print("\n🔍 Checking model access...")
        info = model_info("google/gemma-3-12b-it")
        print(f"✅ Model found: {info.modelId}")
        print(f"   Siblings: {len(info.siblings)} files")
        return True
    except Exception as e:
        print(f"❌ Cannot access model: {e}")
        print("\nAction required:")
        print("  1. Go to: https://huggingface.co/google/gemma-3-12b-it")
        print("  2. Click 'Access Repository'")
        print("  3. Accept Google's license terms")
        print("  4. Wait ~1 minute for approval")
        print("  5. Rerun this script")
        return False

if __name__ == "__main__":
    success = authenticate()
    if success:
        if verify_model_access():
            print("\n" + "=" * 50)
            print("✅ All authenticated! Ready for Phase 2 training")
            print("   Run: python src/train_vision.py")
        else:
            print("\n⚠️  Need to accept model access on HuggingFace")
    else:
        print("\n❌ Authentication failed")
        sys.exit(1)
