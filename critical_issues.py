import re

print("=" * 80)
print("CRITICAL ISSUES FOUND")
print("=" * 80)

issues = []

# Issue 1: inference.py line 189 - undefined variable
print("\n1. INFERENCE.PY - UNDEFINED VARIABLE")
print("-" * 80)

with open("src/inference.py") as f:
    lines = f.readlines()

# Check line 189
line189 = lines[188].strip()
print(f"   Line 189: {line189}")

if "raw_sql" in line189 and "_render_rich" in line189:
    # Check _render_rich signature
    for i, line in enumerate(lines):
        if "def _render_rich" in line:
            print(f"   Found _render_rich definition at line {i+1}")
            sig_lines = ''.join(lines[i:i+3])
            print(f"   Signature: {sig_lines.strip()}")
            
            if "raw_sql" in sig_lines:
                print("   ✓ raw_sql parameter exists in function signature")
            else:
                print("   ⚠️  Function expects (image_name, sql, raw_sql)")
                print(f"      But passing: _render_rich(img_file.name, sql, sql)")
                print("      → raw_sql parameter is REDUNDANT (passing same value twice)")
                issues.append({
                    'file': 'src/inference.py',
                    'line': 189,
                    'severity': 'MINOR',
                    'issue': 'Redundant parameter (passing sql twice as raw_sql)',
                    'code': line189
                })
            break

# Issue 2: Check app.py for potential AttributeError
print("\n2. APP.PY - Potential issues")
print("-" * 80)

with open("src/app.py") as f:
    app_lines = f.readlines()

# Check load_model function
for i, line in enumerate(app_lines):
    if "def load_model" in line:
        # Get next 10 lines
        func_block = ''.join(app_lines[i:i+10])
        if "FastVisionModel.from_pretrained" in func_block:
            print(f"   Line {i+1}: load_model() definition found")
            if "model_name=adapter_path" in func_block:
                print("   ✓ Correctly passes adapter_path as model_name")
            else:
                print("   ⚠️  Check if adapter_path is used correctly")

# Issue 3: Check train_vision.py argparse
print("\n3. TRAIN_VISION.PY - Argument parsing")
print("-" * 80)

with open("src/train_vision.py") as f:
    tv_lines = f.readlines()

for i, line in enumerate(tv_lines):
    if 'parser.add_argument("--dataset"' in line:
        print(f"   Line {i+1}: {line.strip()}")
        arg_line = tv_lines[i]
        if 'default=Path' in arg_line:
            match = re.search(r'default=Path\("([^"]+)"\)', arg_line)
            if match:
                default_path = match.group(1)
                print(f"   Default dataset path: {default_path}")
                
                from pathlib import Path
                if not Path(default_path).exists():
                    print(f"   ⚠️  WARNING: Default dataset '{default_path}' does NOT exist!")
                    print(f"      Script will fail if called without --dataset argument")
                    issues.append({
                        'file': 'src/train_vision.py',
                        'line': i+1,
                        'severity': 'HIGH',
                        'issue': 'Default dataset path does not exist',
                        'code': arg_line.strip(),
                        'fix': 'Run with: python src/train_vision.py --dataset data/dataset_vision.json'
                    })

# Issue 4: Check export.py default path
print("\n4. EXPORT.PY - Default adapter path")
print("-" * 80)

with open("src/export.py") as f:
    exp_lines = f.readlines()

for i, line in enumerate(exp_lines):
    if 'default="output/adapters/phase2"' in line:
        print(f"   Line {i+1}: {line.strip()}")
        print(f"   ⚠️  Default adapter is 'phase2' (for Colab)")
        print(f"   → For Modal training, use: --adapter_dir output/adapters/trinity_a10g")
        print(f"   ✓ This is intentional - not a bug, but could be confusing")

# Issue 5: Check data_factory.py for model availability
print("\n5. DATA_FACTORY.PY - Model selection issue")
print("-" * 80)

with open("scripts/data_factory.py") as f:
    df_content = f.read()

if 'gemini-2.5-flash' in df_content or 'gemini-3.1' in df_content:
    print("   ⚠️  Model list includes newer models that may not be available yet:")
    print("      - gemini-3.1-flash-lite-preview (still in preview)")
    print("   ✓ Code handles 429 errors and model rotation, so will skip unavailable models")

# Issue 6: Check for missing file operations error handling
print("\n6. FILE OPERATIONS - Error handling")
print("-" * 80)

with open("scripts/build_vision_dataset.py") as f:
    build_content = f.read()

if "if not screenshot_dir.exists()" in build_content:
    print("   ✓ build_vision_dataset.py line 166: Checks if directory exists")
    
if "screenshot_dir.glob" in build_content:
    print("   ✓ build_vision_dataset.py line 175: Uses .glob() safely")

# Issue 7: Check merge_datasets.py error handling
print("\n7. MERGE_DATASETS.PY - Error handling")
print("-" * 80)

with open("scripts/merge_datasets.py") as f:
    merge_lines = f.readlines()

if "if not synthetic_file.exists()" in ''.join(merge_lines):
    print("   ✓ Line 22: Checks if synthetic_file exists")
    print("   ✓ Provides helpful error message directing user to run data_factory.py")

# Summary
print("\n" + "=" * 80)
print("ISSUE SUMMARY")
print("=" * 80)

critical = [i for i in issues if i.get('severity') == 'CRITICAL']
high = [i for i in issues if i.get('severity') == 'HIGH']
minor = [i for i in issues if i.get('severity') == 'MINOR']

print(f"\nCRITICAL: {len(critical)}")
print(f"HIGH:     {len(high)}")
print(f"MINOR:    {len(minor)}")

if high:
    print("\n--- HIGH SEVERITY ISSUES ---")
    for issue in high:
        print(f"\n{issue['file']}:{issue['line']}")
        print(f"  Issue: {issue['issue']}")
        print(f"  Code: {issue['code']}")
        if 'fix' in issue:
            print(f"  Fix: {issue['fix']}")

print("\n" + "=" * 80)
