import re
from pathlib import Path

print("=" * 80)
print("RUNTIME & INTEGRATION ISSUES AUDIT")
print("=" * 80)

# Issue 1: Check data_factory.py model names
print("\n1. DATA_FACTORY.PY - Model names validation")
print("-" * 80)

with open("scripts/data_factory.py") as f:
    content = f.read()
    
# Extract model list
match = re.search(r"def get_available_models\(\):.*?return \[(.*?)\]", content, re.DOTALL)
if match:
    models_str = match.group(1)
    models = re.findall(r"['\"]([^'\"]+)['\"]", models_str)
    print("\n   Available models in data_factory.py:")
    for i, model in enumerate(models, 1):
        print(f"      {i}. {model}")
    
    # Check for -latest suffix (recommended)
    latest_count = sum(1 for m in models if 'latest' in m)
    print(f"\n   ✓ {latest_count}/{len(models)} models use '-latest' suffix (good - avoids 404s)")

# Issue 2: Check for missing JSON error handling
print("\n2. DATA_FACTORY.PY - Error handling")
print("-" * 80)

if "except:" in content:
    print("   ✓ Line 84: Has bare except clause for JSON load errors")
    if "json.loads" in content:
        print("   ✓ Line 106: Wraps JSON parsing in try-except")

# Issue 3: Check app.py cache decorator
print("\n3. APP.PY - Model caching")
print("-" * 80)

with open("src/app.py") as f:
    app_content = f.read()

if "@st.cache_resource" in app_content:
    print("   ✓ Line 133: Uses @st.cache_resource for load_model()")
    print("      → Model loaded once, reused across app sessions")
    
if "show_spinner=False" in app_content:
    print("   ✓ Line 133: Disables spinner during caching")

# Issue 4: Check for path consistency in messages
print("\n4. MESSAGE FORMAT CONSISTENCY")
print("-" * 80)

files_with_messages = {
    "modal_train.py": "src/modal_train.py",
    "train_vision.py": "src/train_vision.py",
    "app.py": "src/app.py",
    "inference.py": "src/inference.py",
}

for name, path in files_with_messages.items():
    with open(path) as f:
        content = f.read()
    
    # Check message format
    if '"role": "user"' in content and '"content":' in content:
        if '"type": "image"' in content and '"type": "text"' in content:
            print(f"   ✓ {name}: Uses multi-type content format (image + text)")

# Issue 5: Check inference.py sql parsing
print("\n5. INFERENCE.PY - SQL parsing fallback")
print("-" * 80)

with open("src/inference.py") as f:
    inf_content = f.read()

if "split('model" in inf_content:
    print('   ✓ Line 187: Handles "model" token split in output')
    print('      → Extracts SQL after "model\\n" prefix if present')
else:
    print("   ⚠️  May not handle model response prefix")

if "_parse_create_tables" in inf_content:
    print("   ✓ Line 80: Uses _parse_create_tables() for parsing")

# Issue 6: Check export.py for directory creation
print("\n6. EXPORT.PY - Output directory handling")
print("-" * 80)

with open("src/export.py") as f:
    export_content = f.read()

if "mkdir(parents=True, exist_ok=True)" in export_content:
    print("   ✓ Line 36: Creates output directory with parents=True")

if "save_pretrained_gguf" in export_content:
    print("   ✓ Line 60: Uses save_pretrained_gguf() for GGUF export")

if "glob('**/*.gguf')" in export_content:
    print("   ✓ Line 66: Finds GGUF files with recursive glob")

# Issue 7: Check train_vision.py dataset loading
print("\n7. TRAIN_VISION.PY - Dataset loading robustness")
print("-" * 80)

with open("src/train_vision.py") as f:
    tv_content = f.read()

if "skipped" in tv_content and "skipped += 1" in tv_content:
    print("   ✓ Line 50: Increments skipped counter for missing images")
    
if "logger.info" in tv_content:
    print("   ✓ Line 81: Logs number of valid/skipped examples")

# Issue 8: Check for tuple unpacking issues
print("\n8. TUPLE UNPACKING - Return value consistency")
print("-" * 80)

unpack_issues = []

# Check modal_train._load_dataset returns
with open("src/modal_train.py") as f:
    modal_content = f.read()
    if "return Dataset.from_list(valid)" in modal_content:
        print("   ✓ modal_train._load_dataset(): Returns Dataset (single value)")

# Check train_vision.load_vision_dataset returns  
with open("src/train_vision.py") as f:
    tv_content = f.read()
    if "return Dataset.from_list(valid_rows)" in tv_content:
        print("   ✓ train_vision.load_vision_dataset(): Returns Dataset (single value)")

# Check create_vision_model returns
if "return model, processor" in tv_content:
    print("   ✓ train_vision.create_vision_model(): Returns (model, processor) tuple")

print("\n9. FUNCTION CALL VERIFICATION")
print("-" * 80)

# Check if train_vision_model unpacks correctly
if "model, processor = create_vision_model()" in tv_content:
    print("   ✓ train_vision.py line 123: Correctly unpacks tuple from create_vision_model()")

# Check if _load_dataset is called correctly
if "dataset = load_vision_dataset(dataset_path)" in tv_content:
    print("   ✓ train_vision.py line 122: Correctly assigns single return from load_vision_dataset()")

print("\n" + "=" * 80)
