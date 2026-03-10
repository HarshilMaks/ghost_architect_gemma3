# Ghost Architect - Quick Fix Guide

## 🔴 CRITICAL: Fix These NOW

### Fix #1: train_vision.py Line 169
**File:** `src/train_vision.py`  
**Current:**
```python
parser.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
```

**Change to:**
```python
parser.add_argument("--dataset", type=Path, default=Path("data/dataset_vision.json"))
```

**Why:** File `data/dataset.json` doesn't exist. Actual file is `data/dataset_vision.json`.

---

### Fix #2: inference.py Lines 62 and 189
**File:** `src/inference.py`

**Line 62 - Change from:**
```python
def _render_rich(image_name: str, sql: str, raw_sql: str):
```

**Change to:**
```python
def _render_rich(image_name: str, sql: str):
```

**Line 189 - Change from:**
```python
    _render_rich(img_file.name, sql, sql)
```

**Change to:**
```python
    _render_rich(img_file.name, sql)
```

**Why:** The `raw_sql` parameter is never used. Passing it twice is a code smell.

---

## 🟠 HIGH: Important Improvements

### Fix #3: Add Documentation to modal_train.py
**File:** `src/modal_train.py` - Line 256 (above `upload_dataset` function)

**Add:**
```python
@app.local_entrypoint()
def upload_dataset(dataset_filename: str = "dataset_vision.json"):
    """
    Upload dataset JSON + ui_screenshots/ to Modal volumes.
    
    ⚠️  IMPORTANT: Must be run from project root directory!
    
    Usage:
        modal run src/modal_train.py::upload_dataset
        modal run src/modal_train.py::upload_dataset --dataset-filename dataset_merged.json
    
    Do NOT run this from a subdirectory.
    """
```

**Why:** Script uses relative paths and will fail if run from wrong directory.

---

### Fix #4: Add Path Validation to train_vision.py
**File:** `src/train_vision.py` - Add before line 171 (after argparse)

**Add:**
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset_vision.json"))
    parser.add_argument("--output", type=str, default="output/adapters/vision_trinity")
    args = parser.parse_args()
    
    # NEW: Validate dataset exists
    if not args.dataset.exists():
        print(f"\n❌ ERROR: Dataset not found: {args.dataset}")
        print(f"   Make sure you're in the project root directory")
        print(f"\n   Example:\n      cd /path/to/ghost_architect_gemma3")
        print(f"      python src/train_vision.py --dataset data/dataset_vision.json\n")
        import sys
        sys.exit(1)
    
    train_vision_model(args.dataset, args.output)
```

**Why:** Better error messages for users.

---

## 🟡 MEDIUM: Nice-to-Have Improvements

### Fix #5: Auto-detect Adapter in app.py
**File:** `src/app.py` - Lines 35-40

**Change from:**
```python
adapter_dir = st.text_input(
    "Adapter path",
    value="output/adapters/trinity_a10g",
    help="Path to your trained LoRA adapter directory",
)
st.caption("Change this to `output/adapters/phase2` if you trained on Colab.")
```

**Change to:**
```python
from pathlib import Path

# Auto-detect which adapter exists
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

**Why:** Better UX - automatically finds the right adapter.

---

### Fix #6: Update export.py Help Text
**File:** `src/export.py` - Line 81-82

**Change from:**
```python
parser.add_argument("--adapter_dir", type=str, default="output/adapters/phase2",
                    help="Directory containing saved LoRA adapter weights")
```

**Change to:**
```python
parser.add_argument("--adapter_dir", type=str, default="output/adapters/phase2",
                    help=(
                        "Directory containing saved LoRA adapter weights\n"
                        "  Modal training:  output/adapters/trinity_a10g\n"
                        "  Colab training:  output/adapters/phase2\n"
                        "  Local training:  output/adapters/vision_trinity"
                    ))
```

**Why:** Clarifies which path to use for different training methods.

---

## 📋 Checklist

### Implement these changes:
- [ ] Fix train_vision.py line 169 (dataset path)
- [ ] Fix inference.py lines 62 & 189 (remove unused parameter)
- [ ] Add docstring to modal_train.py line 256
- [ ] Add validation to train_vision.py main()
- [ ] Enhance app.py adapter detection (optional)
- [ ] Update export.py help text (optional)

### Testing after fixes:
```bash
# Test train_vision.py works with default args
python src/train_vision.py --help

# Test inference.py runs without errors
python src/inference.py --help

# Check no import errors
python -c "from src import modal_train, train_vision, app, inference, export"
```

---

## Time Estimate
- **Critical fixes:** 5 minutes
- **High priority fixes:** 10 minutes  
- **Nice-to-have:** 15 minutes
- **Total:** ~30 minutes

---

**Status:** After these fixes, project is fully deployable.
