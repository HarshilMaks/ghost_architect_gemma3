import re
from pathlib import Path

print("=" * 80)
print("HARDCODED PATHS & ENVIRONMENT VARIABLES AUDIT")
print("=" * 80)

files_to_check = [
    "src/modal_train.py",
    "src/train_vision.py", 
    "src/app.py",
    "src/inference.py",
    "src/export.py",
    "scripts/data_factory.py",
    "scripts/merge_datasets.py",
    "scripts/build_vision_dataset.py",
]

print("\n1. HARDCODED PATHS")
print("-" * 80)

hardcoded_issues = []

for filepath in files_to_check:
    with open(filepath) as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        # Look for hardcoded paths
        if re.search(r'(Path|"data/|"output/|"/data|"/output)', line) and '#' not in line.split('"')[0]:
            if 'data/' in line or 'output/' in line or '/dataset' in line:
                # Skip comments and docstrings
                if not line.strip().startswith('#'):
                    path_match = re.search(r'["\'](.*?)["\']', line)
                    if path_match:
                        path = path_match.group(1)
                        if any(x in path for x in ['data/', 'output/', '/dataset', '/cache', '/hf-cache']):
                            hardcoded_issues.append({
                                'file': filepath,
                                'line': i,
                                'path': path,
                                'code': line.strip()[:70]
                            })

if hardcoded_issues:
    for issue in hardcoded_issues[:15]:  # Show first 15
        print(f"\n   {issue['file']}:{issue['line']}")
        print(f"      Path: {issue['path']}")
        print(f"      Code: {issue['code']}")
else:
    print("   ✓ No suspicious hardcoded paths found")

print("\n2. ENVIRONMENT VARIABLES")
print("-" * 80)

env_vars_needed = {}

for filepath in files_to_check:
    with open(filepath) as f:
        content = f.read()
    
    # Find os.environ or os.getenv calls
    env_matches = re.findall(r'os\.environ\[?"(\w+)"?\]|os\.getenv\("(\w+)"', content)
    for match in env_matches:
        var_name = match[0] or match[1]
        if var_name not in env_vars_needed:
            env_vars_needed[var_name] = []
        env_vars_needed[var_name].append(filepath)

for var, files in sorted(env_vars_needed.items()):
    print(f"\n   ${var}")
    print(f"      Required by: {', '.join(set(files))}")
    
    # Check if in .env or .env.example
    with open('.env.example') as f:
        example = f.read()
    with open('.env') as f:
        dotenv = f.read()
    
    if var in example:
        print(f"      ✓ Documented in .env.example")
    else:
        print(f"      ⚠️  NOT in .env.example")
        
    if var in dotenv:
        print(f"      ✓ Present in .env")
    else:
        print(f"      ⚠️  NOT in .env (must be set before running)")

print("\n3. PATH VALIDATION - Do referenced paths exist?")
print("-" * 80)

critical_paths = [
    "data/dataset_vision.json",
    "data/ui_screenshots",
    "configs/training_config.yaml",
    "output/adapters",
]

for path_str in critical_paths:
    p = Path(path_str)
    if p.exists():
        if p.is_dir():
            print(f"   ✓ {path_str}/ EXISTS (directory)")
        else:
            print(f"   ✓ {path_str} EXISTS (file)")
    else:
        print(f"   ❌ {path_str} - NOT FOUND")

print("\n4. RELATIVE PATH DEPENDENCIES")
print("-" * 80)

print("""
   The following scripts use relative paths and MUST be run from project root:
   
   • scripts/merge_datasets.py
   • scripts/build_vision_dataset.py  
   • scripts/data_factory.py
   
   These should be called as:
   • cd /home/harshil/ghost_architect_gemma3
   • python scripts/merge_datasets.py
   
   NOT from subdirectories.
""")

print("=" * 80)
