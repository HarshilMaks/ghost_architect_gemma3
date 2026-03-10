# Ghost Architect - Detailed Code Audit Issues

## Issue #1: CRITICAL - train_vision.py - Wrong Default Dataset Path

**File:** `src/train_vision.py`  
**Line:** 169  
**Severity:** CRITICAL

### Code
```python
parser.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
```

### Problem
- The default path is `data/dataset.json` but this file does **NOT exist**
- The actual file is `data/dataset_vision.json`
- If user runs without explicit `--dataset` flag, script **WILL FAIL**

### Evidence
```bash
$ ls -la data/dataset*.json
-rw-r--r-- data/dataset_merged.json  (5287 items)
-rw-r--r-- data/dataset_vision.json  (287 items)
# data/dataset.json does NOT exist
```

### Expected Error
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/dataset.json'
```

### Current Workaround
```bash
python src/train_vision.py --dataset data/dataset_vision.json
```

### Fix Required
Change line 169 to:
```python
parser.add_argument("--dataset", type=Path, default=Path("data/dataset_vision.json"))
```

---

## Issue #2: MEDIUM - inference.py - Unused Function Parameter

**File:** `src/inference.py`  
**Lines:** 62, 189  
**Severity:** MEDIUM (works but code smell)

### Code
```python
# Line 62 - Function definition
def _render_rich(image_name: str, sql: str, raw_sql: str):
    # ... function body ...
    console.print(Panel(
        f"[white]{sql}[/white]",  # Line 85: Uses 'sql'
        # raw_sql is NEVER used
    ))

# Line 189 - Function call
_render_rich(img_file.name, sql, sql)  # Passes sql twice
```

### Problem
- Function signature accepts 3 parameters: `image_name`, `sql`, `raw_sql`
- Call passes `sql` for both the `sql` and `raw_sql` parameters
- The `raw_sql` parameter is **never referenced** in the function body
- Indicates incomplete refactoring or leftover debug code

### Function Body Analysis
```python
def _render_rich(image_name: str, sql: str, raw_sql: str):
    console.print(Panel(
        f"[bold cyan]Ghost Architect[/bold cyan] — Schema Analysis\n"
        f"[dim]Image:[/dim] [yellow]{image_name}[/yellow]",  # ✓ Uses image_name
        ...
    ))
    
    tables = _parse_create_tables(sql)  # ✓ Uses sql
    
    if not tables:
        console.print(Panel(
            f"[white]{sql}[/white]",  # ✓ Uses sql
            ...
        ))
        return
    
    # raw_sql is never used anywhere ❌
```

### Why This Matters
- Confuses future maintainers about intent
- May indicate incomplete implementation
- Takes up mental bandwidth when reading code

### Fix Options

**Option A: Remove unused parameter (preferred)**
```python
# Line 62
def _render_rich(image_name: str, sql: str):
    # Function body stays identical
    ...

# Line 189
_render_rich(img_file.name, sql)
```

**Option B: Use the parameter (if intended)**
```python
# If raw_sql should be shown as fallback:
def _render_rich(image_name: str, sql: str, raw_sql: str):
    # Show parsed tables if available, else show raw
    tables = _parse_create_tables(sql)
    if not tables:
        console.print(Panel(f"[white]{raw_sql}[/white]", ...))  # Use raw_sql
    ...
```

---

## Issue #3: HIGH - modal_train.py - Relative Paths in Modal Context

**File:** `src/modal_train.py`  
**Lines:** 265-286  
**Severity:** HIGH (can cause upload failures)

### Code
```python
@app.local_entrypoint()
def upload_dataset(dataset_filename: str = "dataset_vision.json"):
    """Upload dataset JSON + ui_screenshots/ to Modal."""
    import os

    local_json = Path("data") / dataset_filename
    
    # Check both root data/ and synthetic_factory/ subfolder
    if not local_json.exists():
        local_json = Path("data/synthetic_factory/synthetic_dataset.json")

    local_screenshots = Path("data/ui_screenshots")
    local_synthetic_screenshots = Path("data/synthetic_factory/screenshots")

    assert local_json.exists(), f"Missing {local_json}"
```

### Problem
- Uses **relative paths** to locate local files
- Will fail if run from any directory other than project root
- No validation that working directory is correct

### Expected Failure Scenario
```bash
$ cd src
$ modal run modal_train.py::upload_dataset
# Error: AssertionError: Missing data/dataset_vision.json
# (Correct - we're in src/, not project root)
```

### Why It Matters
- Confuses users who may not understand relative path requirements
- No clear error message about working directory
- Works by accident when run via `modal run` (which changes directory)

### Current Behavior
- ✓ Works: `modal run src/modal_train.py::upload_dataset`
- ✓ Works: `cd /project && python -c "from src.modal_train import upload_dataset; upload_dataset()"`
- ❌ Fails: Direct import from wrong directory

### Recommendation
Add to docstring:
```python
def upload_dataset(dataset_filename: str = "dataset_vision.json"):
    """
    Upload dataset JSON + ui_screenshots/ to Modal.
    
    Must be run from project root directory!
    
    Examples:
        modal run src/modal_train.py::upload_dataset
        modal run src/modal_train.py::upload_dataset --dataset-filename dataset_merged.json
    """
```

---

## Issue #4: MEDIUM - app.py - Hardcoded Adapter Path

**File:** `src/app.py`  
**Line:** 37  
**Severity:** MEDIUM (UX friction)

### Code
```python
with st.sidebar:
    st.title("👻 Ghost Architect")
    st.markdown("**UI → PostgreSQL Schema**")
    st.divider()

    adapter_dir = st.text_input(
        "Adapter path",
        value="output/adapters/trinity_a10g",  # Line 37
        help="Path to your trained LoRA adapter directory",
    )
    st.caption("Change this to `output/adapters/phase2` if you trained on Colab.")
```

### Problem
- Default adapter: `trinity_a10g` (created by Modal training)
- Alternative: `phase2` (created by Colab training)
- User must manually change if they used Colab
- Confusing for first-time users

### Available Adapters
```
output/adapters/
├── phase1/       (legacy - old Colab run)
├── phase2/       (Colab training output)
└── trinity_a10g/ (Modal training output - currently exists)
```

### Improvement
Auto-detect which adapter exists:
```python
# Determine default adapter path
default_adapter = "output/adapters/trinity_a10g"
if not Path(default_adapter).exists():
    if Path("output/adapters/phase2").exists():
        default_adapter = "output/adapters/phase2"
    elif Path("output/adapters/phase1").exists():
        default_adapter = "output/adapters/phase1"

adapter_dir = st.text_input(
    "Adapter path",
    value=default_adapter,
    help="Path to your trained LoRA adapter directory (auto-detected)",
)
```

---

## Issue #5: MEDIUM - export.py - Mismatched Default Adapter

**File:** `src/export.py`  
**Line:** 81  
**Severity:** MEDIUM (documentation issue)

### Code
```python
parser.add_argument(
    "--adapter_dir", 
    type=str, 
    default="output/adapters/phase2",  # Line 81
    help="Directory containing saved LoRA adapter weights"
)
```

### Problem
- Default is `phase2` (for Colab training)
- But Modal training creates `trinity_a10g`
- After Modal training, user must specify: `--adapter_dir output/adapters/trinity_a10g`

### Training Output Paths
- **Colab:** `output/adapters/phase2`
- **Modal:** `output/adapters/trinity_a10g`
- **Local:** `output/adapters/vision_trinity`

### Why Default is Colab
- Legacy code from Colab-first implementation
- Modal training added later
- Different training environments produce differently-named outputs

### Mitigation
Add to help text:
```python
parser.add_argument(
    "--adapter_dir",
    type=str,
    default="output/adapters/phase2",
    help=(
        "Directory containing saved LoRA adapter weights\n"
        "• Modal training: output/adapters/trinity_a10g\n"
        "• Colab training: output/adapters/phase2\n"
        "• Local training: output/adapters/vision_trinity"
    )
)
```

---

## Issue #6: MEDIUM - train_vision.py - No Path Validation

**File:** `src/train_vision.py`  
**Lines:** 169-173  
**Severity:** MEDIUM (poor error messages)

### Code
```python
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
parser.add_argument("--output", type=str, default="output/adapters/vision_trinity")
args = parser.parse_args()

train_vision_model(args.dataset, args.output)
```

### Problem
- No validation that `--dataset` path exists
- Script silently attempts to open non-existent file
- User gets generic Python error with no context

### Failure Scenario 1: Missing File
```bash
$ python src/train_vision.py
FileNotFoundError: [Errno 2] No such file or directory: 'data/dataset.json'
# User doesn't know this is the default dataset.json issue
# Or that they should use dataset_vision.json
```

### Failure Scenario 2: Wrong Working Directory
```bash
$ cd src && python train_vision.py --dataset data/dataset_vision.json
FileNotFoundError: [Errno 2] No such file or directory: 'data/dataset_vision.json'
# Script runs from src/, but dataset is at ../data/
# Error message doesn't explain this
```

### Recommended Fix
```python
def main(args):
    if not args.dataset.exists():
        print(f"\n❌ ERROR: Dataset not found at {args.dataset}")
        print(f"\nMake sure you're in the project root directory.")
        print(f"\nExample usage:")
        print(f"  cd /path/to/ghost_architect_gemma3")
        print(f"  python src/train_vision.py --dataset data/dataset_vision.json")
        sys.exit(1)
    
    if not args.dataset.is_file():
        print(f"\n❌ ERROR: {args.dataset} exists but is not a file")
        sys.exit(1)
    
    train_vision_model(args.dataset, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset_vision.json"))
    parser.add_argument("--output", type=str, default="output/adapters/vision_trinity")
    args = parser.parse_args()
    main(args)
```

---

## Issue #7: LOW - data_factory.py - Preview Models

**File:** `scripts/data_factory.py`  
**Lines:** 49-58  
**Severity:** LOW (gracefully handled)

### Code
```python
def get_available_models():
    """Return verified list of models based on user's check_models output."""
    return [
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-flash-lite-latest',
        'gemini-flash-latest',
        'gemini-pro-latest',
        'gemini-3.1-flash-lite-preview',  # ⚠️ Preview
        'gemini-2.0-flash-lite-001',
        'gemini-2.5-pro'
    ]
```

### Problem
- Some models may not be available yet
- `gemini-3.1-flash-lite-preview` is in preview status
- `gemini-2.5-pro` may not be available to all users

### Current Handling
- ✓ Code catches 429 errors (quota exceeded)
- ✓ Auto-rotates to next model on error
- ✓ Logs which model failed
- ✓ Continues gracefully

### Status
**NOT A BUG** - The error handling works correctly. This is intentional fallback behavior.

### Recommendation
Only a documentation note needed:
```python
def get_available_models():
    """
    Return list of Gemini models.
    
    Some models may be in preview and not available.
    Script handles unavailable models gracefully via quota rotation.
    """
```

---

## Issue #8: LOW - build_vision_dataset.py - Silent Skipping

**File:** `scripts/build_vision_dataset.py`  
**Line:** 189  
**Severity:** LOW (acceptable)

### Code
```python
for idx, img_path in enumerate(all_images, 1):
    # Progress indicator
    if idx % 10 == 0:
        print(f"  ✓ Processing {idx}/{len(all_images)}...", end='\r')
    
    # Validate image
    if not validate_image(img_path):
        invalid_count += 1
        continue  # Line 191: Silent continue
```

### Current Behavior
```python
def validate_image(image_path: Path) -> bool:
    """Check if image is valid and not corrupted"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"  ⚠️  Invalid image: {image_path.name} - {e}")  # Error IS printed
        return False
```

### Assessment
- ✓ Invalid images ARE logged by validate_image()
- ✓ Count is tracked and reported in summary
- ✓ Behavior is correct and transparent

### Status
**NO ACTION NEEDED** - This is working as intended.

---

## Summary of All Issues

| # | File | Line | Issue | Severity | Status |
|---|------|------|-------|----------|--------|
| 1 | train_vision.py | 169 | Wrong default dataset path | CRITICAL | Needs fix |
| 2 | inference.py | 62, 189 | Unused function parameter | MEDIUM | Needs fix |
| 3 | modal_train.py | 265-286 | Relative paths in upload | HIGH | Needs docs |
| 4 | app.py | 37 | Hardcoded adapter path | MEDIUM | Suggestion |
| 5 | export.py | 81 | Mismatched default | MEDIUM | Needs docs |
| 6 | train_vision.py | 169-173 | No path validation | MEDIUM | Enhancement |
| 7 | data_factory.py | 49-58 | Preview models | LOW | N/A (works) |
| 8 | build_vision_dataset.py | 189 | Silent skipping | LOW | N/A (works) |

---

## Implementation Priority

### Phase 1: Critical (Do immediately)
1. Fix train_vision.py line 169 - dataset path
2. Remove unused parameter from inference.py

### Phase 2: High Priority (Before production)
3. Add validation to train_vision.py
4. Document modal_train.py working directory requirement
5. Enhance app.py adapter detection

### Phase 3: Nice-to-have (Polish)
6. Update help text in export.py
7. Standardize adapter naming
8. Add verbose logging

---

**Audit Date:** March 10, 2025  
**Status:** Ready for deployment after Phase 1 fixes
