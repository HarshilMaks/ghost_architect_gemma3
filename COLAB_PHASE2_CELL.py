# Colab Cell: Phase 2 HuggingFace Authentication
# Add this cell to your Colab notebook BEFORE running train_vision.py

# ============================================================================
# Cell: Mount Drive & Authenticate HuggingFace
# ============================================================================

from google.colab import drive
import os

drive.mount("/content/drive", force_remount=True)

PROJECT_DIR = '/content/ghost_architect_gemma3'
DRIVE_COPY = '/content/drive/MyDrive/ghost_architect_gemma3'
REPO_URL = 'https://github.com/HarshilMaks/ghost_architect_gemma3.git'

# Sync repository
if os.path.exists(f'{DRIVE_COPY}/src/train_vision.py'):
    print("Using repo from Google Drive...")
    !rm -rf {PROJECT_DIR}
    !cp -r {DRIVE_COPY} {PROJECT_DIR}
elif REPO_URL.strip():
    print("Cloning from GitHub...")
    !rm -rf {PROJECT_DIR}
    !git clone {REPO_URL} {PROJECT_DIR}
else:
    print("Using local project...")

%cd {PROJECT_DIR}

print("\n✅ Repository synced")

# ============================================================================
# Install dependencies
# ============================================================================

!pip install -q transformers peft bitsandbytes accelerate datasets pillow python-dotenv huggingface-hub

# ============================================================================
# Authenticate with HuggingFace
# ============================================================================

from huggingface_hub import login, model_info
from dotenv import load_dotenv
import os

# Load .env if it exists in Drive
env_path = "/content/drive/MyDrive/ghost_architect_gemma3/.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("🔐 Found HUGGINGFACE_TOKEN in .env")
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Authenticated with HuggingFace")
    else:
        print("⚠️  .env exists but HUGGINGFACE_TOKEN not set")
        print("Using interactive login...")
        from huggingface_hub import notebook_login
        notebook_login()
else:
    print("⚠️  .env not found on Drive")
    print("Using interactive login...")
    from huggingface_hub import notebook_login
    notebook_login()

# Verify access to model
try:
    info = model_info("google/gemma-3-12b-it")
    print(f"✅ Access to google/gemma-3-12b-it confirmed!")
except Exception as e:
    print(f"❌ Cannot access model: {e}")
    print("Please:")
    print("  1. Go to https://huggingface.co/google/gemma-3-12b-it")
    print("  2. Click 'Access Repository' and accept terms")
    print("  3. Then run this cell again")

# ============================================================================
# Now you can run Phase 2 training
# ============================================================================
# !python src/train_vision.py --dataset data/dataset.json
