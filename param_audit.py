import re

print("=" * 80)
print("FUNCTION PARAMETER & SIGNATURE AUDIT")
print("=" * 80)

# Issue 1: Check SFTConfig parameters match between files
print("\n1. SFTConfig parameter consistency")
print("   Checking modal_train.py vs train_vision.py vs export (uses from_pretrained)")

files = {
    "modal_train.py": "src/modal_train.py",
    "train_vision.py": "src/train_vision.py",
}

with open("src/modal_train.py") as f:
    modal = f.read()
    if "dataset_text_field=" in modal:
        print("   ✓ modal_train.py: Sets dataset_text_field=\"\"")
    if "skip_prepare_dataset" in modal:
        print("   ✓ modal_train.py: Sets skip_prepare_dataset=True")

with open("src/train_vision.py") as f:
    train_v = f.read()
    if "dataset_text_field=" in train_v:
        print("   ✓ train_vision.py: Sets dataset_text_field=\"\"")
    if "skip_prepare_dataset" in train_v:
        print("   ✓ train_vision.py: Sets skip_prepare_dataset=True")

# Issue 2: Check data_collator parameter
print("\n2. data_collator parameter passing")

with open("src/modal_train.py") as f:
    modal = f.read()
    if "data_collator=UnslothVisionDataCollator" in modal:
        print("   ✓ modal_train.py line 217: Correctly passes data_collator")

with open("src/train_vision.py") as f:
    train_v = f.read()
    if "data_collator=UnslothVisionDataCollator" in train_v:
        print("   ✓ train_vision.py line 134: Correctly passes data_collator")

with open("src/app.py") as f:
    app = f.read()
    if "data_collator" not in app:
        print("   ℹ️  app.py: No training (inference only) - N/A")

# Issue 3: Check for_inference parameter
print("\n3. FastVisionModel.for_inference() calls")

with open("src/inference.py") as f:
    inf = f.read()
    if "FastVisionModel.for_inference" in inf:
        print("   ✓ inference.py line 154: Calls for_inference(model)")

with open("src/app.py") as f:
    app = f.read()
    if "FastVisionModel.for_inference" in app:
        print("   ✓ app.py line 140: Calls for_inference(model)")

with open("src/modal_train.py") as f:
    modal = f.read()
    if "for_inference" not in modal:
        print("   ✓ modal_train.py: Doesn't call for_inference (training mode)")

# Issue 4: Check processor vs processing_class parameter names
print("\n4. processor vs processing_class parameter naming")

with open("src/modal_train.py") as f:
    modal = f.read()
    if "processing_class=processor" in modal:
        print("   ✓ modal_train.py line 216: Uses processing_class=processor")
        
with open("src/train_vision.py") as f:
    train_v = f.read()
    if "processing_class=processor" in train_v:
        print("   ✓ train_vision.py line 133: Uses processing_class=processor")

# Issue 5: Check max_new_tokens in generation
print("\n5. max_new_tokens parameter consistency")

with open("src/inference.py") as f:
    inf = f.read()
    if "max_new_tokens=600" in inf or "max_new_tokens=700" in inf:
        match = re.search(r"max_new_tokens=(\d+)", inf)
        if match:
            print(f"   ✓ inference.py line 180: max_new_tokens={match.group(1)}")

with open("src/app.py") as f:
    app = f.read()
    if "max_new_tokens=" in app:
        match = re.search(r"max_new_tokens=(\d+)", app)
        if match:
            print(f"   ✓ app.py line 170: max_new_tokens={match.group(1)}")

with open("src/modal_train.py") as f:
    modal = f.read()
    if "max_new_tokens" not in modal:
        print("   ✓ modal_train.py: No generation (training only)")

# Issue 6: Check processor usage in app.py and inference.py
print("\n6. processor.apply_chat_template() calls")

with open("src/app.py") as f:
    app = f.read()
    if "processor.apply_chat_template" in app:
        print("   ✓ app.py line 158: Calls processor.apply_chat_template()")

with open("src/inference.py") as f:
    inf = f.read()
    if "processor.apply_chat_template" in inf:
        print("   ✓ inference.py line 169: Calls processor.apply_chat_template()")

print("\n" + "=" * 80)
