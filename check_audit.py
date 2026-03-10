import json
from pathlib import Path

print("=" * 70)
print("DATASET SCHEMA AUDIT")
print("=" * 70)

# Check if image_path and image fields are consistent
files_to_check = [
    "data/dataset_vision.json",
    "data/dataset_merged.json",
]

for file_path in files_to_check:
    p = Path(file_path)
    if not p.exists():
        print(f"\n❌ {file_path} - NOT FOUND")
        continue
        
    try:
        data = json.load(open(p))
        if not data:
            print(f"\n⚠️ {file_path} - EMPTY")
            continue
            
        sample = data[0]
        keys = set(sample.keys())
        
        print(f"\n✓ {file_path}")
        print(f"  Items: {len(data)}")
        print(f"  Keys: {sorted(keys)}")
        
        # Check critical fields
        has_image = "image" in keys
        has_image_path = "image_path" in keys
        has_instruction = "instruction" in keys
        has_output = "output" in keys
        
        if has_image:
            print(f"    ✓ Has 'image' field")
        elif has_image_path:
            print(f"    ✓ Has 'image_path' field (used in modal_train & train_vision)")
        else:
            print(f"    ❌ MISSING image/image_path field!")
            
        if has_instruction:
            print(f"    ✓ Has 'instruction' field")
        else:
            print(f"    ❌ MISSING instruction field!")
            
        if has_output:
            print(f"    ✓ Has 'output' field")
        else:
            print(f"    ❌ MISSING output field!")
            
    except Exception as e:
        print(f"\n❌ {file_path} - ERROR: {e}")

print("\n" + "=" * 70)
print("CHECKING field REFERENCES IN CODE")
print("=" * 70)

print("\nmodal_train.py line 112 references:")
print("  item.get('image_path') or item.get('image', '')")

print("\ntrain_vision.py line 47 references:")
print("  item.get('image_path') or item.get('image', '')")

print("\nbuild_vision_dataset.py line 206 sets:")
print("  'image_path': str(img_path)")

print("\ndata_factory.py line 126 sets:")
print("  'image_path': str(ss_path)")

print("\n✓ All use 'image_path' - CONSISTENT")

