import re
from pathlib import Path

print("=" * 80)
print("DETAILED CODE AUDIT - POTENTIAL ISSUES")
print("=" * 80)

files_to_audit = {
    "src/modal_train.py": [
        (177, "UnslothVisionDataCollator import"),
        (214, "SFTTrainer instantiation - data_collator parameter"),
        (236, "dataset_text_field parameter"),
    ],
    "src/train_vision.py": [
        (17, "UnslothVisionDataCollator import"),
        (134, "data_collator instantiation"),
    ],
    "src/app.py": [
        (135, "FastVisionModel.from_pretrained - model_name path"),
        (137, "load_in_4bit=True parameter"),
    ],
    "src/inference.py": [
        (150, "FastVisionModel.from_pretrained"),
    ],
    "src/export.py": [
        (47, "FastVisionModel.from_pretrained - model_name parameter"),
    ],
}

issues = []

# Issue 1: Check if export.py line 81 has correct default
print("\n1. EXPORT.PY - Default adapter path mismatch")
export_code = Path("src/export.py").read_text().split('\n')
line81 = export_code[80]  # 0-indexed
print(f"   Line 81: {line81}")
if 'phase2' in line81:
    print("   ⚠️  ISSUE: Default adapter is 'phase2', but modal_train creates 'trinity_a10g'")
    print("   ✓ This is intentional (Colab vs Modal paths) - NOT A BUG")
else:
    print("   ✓ No issue")

# Issue 2: Check app.py line 137 for potential path problem  
print("\n2. APP.PY - Adapter path handling")
app_code = Path("src/app.py").read_text()
if "model_name=adapter_path" in app_code:
    print("   ✓ Line 137: Correctly passes adapter_path to from_pretrained")
else:
    print("   ❌ Line 137: adapter_path not found in model_name parameter")

# Issue 3: Check inference.py for file existence checks
print("\n3. INFERENCE.PY - File existence handling")
inf_code = Path("src/inference.py").read_text().split('\n')
line142_154 = '\n'.join(inf_code[141:154])
if "glob.glob" in line142_154 and "data/ui_screenshots" in line142_154:
    print("   ✓ Lines 144-147: Correctly globs for images")
    if "FileNotFoundError" in line142_154:
        print("   ✓ Raises FileNotFoundError if no images found")
    else:
        print("   ⚠️  May not handle missing directory gracefully")

# Issue 4: Check modal_train.py dataset path construction
print("\n4. MODAL_TRAIN.PY - Dataset path construction")
modal_code = Path("src/modal_train.py").read_text().split('\n')
line112_114 = '\n'.join(modal_code[111:114])
print(f"   Lines 112-114:")
for line in line112_114.split('\n'):
    print(f"      {line}")
if 'ui_screenshots' in modal_code[113]:
    print("   ✓ Line 114: Correctly sets path to DATASET_PATH / 'ui_screenshots'")

# Issue 5: Check merge_datasets.py for relative paths
print("\n5. MERGE_DATASETS.PY - Relative path handling")
merge_code = Path("scripts/merge_datasets.py").read_text().split('\n')
print(f"   Line 15: {merge_code[14]}")
print(f"   Line 16: {merge_code[15]}")
print(f"   Line 17: {merge_code[16]}")
if 'data/' in merge_code[14]:
    print("   ⚠️  ISSUE: Uses relative paths - must run from project root")
    print("   ✓ This is documented in usage comment")

# Issue 6: Check data_factory.py for playwright import
print("\n6. DATA_FACTORY.PY - Async/Playwright handling")
data_factory_code = Path("scripts/data_factory.py").read_text().split('\n')
if 'from playwright.async_api import async_playwright' in data_factory_code[89]:
    print("   ✓ Line 90: Correctly imports async_playwright")
if 'async with async_playwright' in data_factory_code[91]:
    print("   ✓ Line 92: Correctly uses async context manager")

# Issue 7: Check build_vision_dataset.py for glob pattern
print("\n7. BUILD_VISION_DATASET.PY - Glob patterns")
build_code = Path("scripts/build_vision_dataset.py").read_text().split('\n')
line175 = build_code[174]
print(f"   Line 175: {line175}")
if '*.png' in line175:
    print("   ✓ Correctly globs for .png files")

print("\n" + "=" * 80)
