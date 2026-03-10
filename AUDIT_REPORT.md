# Ghost Architect Code Audit Report

## Executive Summary
- **Total Files Audited:** 9 Python files + 1 YAML config
- **Critical Issues:** 2
- **High-Priority Issues:** 2
- **Medium Issues:** 4
- **Low Issues (Minor/Informational):** 3
- **Overall Status:** MOSTLY SOUND with documented workarounds

---

## CRITICAL ISSUES

### Issue #1: TRAIN_VISION.PY - Incorrect Default Dataset Path
**File:** `src/train_vision.py`  
**Line:** 169  
**Severity:** HIGH  

```python
parser.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
```

**Problem:**
- Default path is `data/dataset.json` but the actual file is `data/dataset_vision.json`
- Script will fail if run without explicit `--dataset` argument
- Error: `FileNotFoundError: [Errno 2] No such file or directory: 'data/dataset.json'`

**Fix Required:**
```python
parser.add_argument("--dataset", type=Path, default=Path("data/dataset_vision.json"))
```

**Current Workaround:**
Users must explicitly provide: `python src/train_vision.py --dataset data/dataset_vision.json`

---

### Issue #2: INFERENCE.PY - Redundant Parameter in Function Call
**File:** `src/inference.py`  
**Line:** 189  
**Severity:** MEDIUM (works but incorrect)

```python
_render_rich(img_file.name, sql, sql)
```

**Problem:**
- Function signature expects three distinct parameters: `image_name, sql, raw_sql`
- Call passes `sql` twice (for both `sql` and `raw_sql` parameters)
- The `raw_sql` parameter is never actually used in `_render_rich()` function body
- Indicates incomplete refactoring or dead code

**Function Signature (line 62):**
```python
def _render_rich(image_name: str, sql: str, raw_sql: str):
```

**Actual Usage in Function:**
- Line 80: `tables = _parse_create_tables(sql)` - only `sql` is used
- Line 85: `f"[white]{sql}[/white]"` - only `sql` is used
- `raw_sql` parameter is **never referenced**

**Recommended Fix:**
Either remove the unused parameter:
```python
def _render_rich(image_name: str, sql: str):
    # ... rest of code stays same
```

Or fix the call to pass different values if intended:
```python
_render_rich(img_file.name, sql, full_text)  # if raw_sql should be full_text
```

---

## HIGH PRIORITY ISSUES

### Issue #3: MODAL_TRAIN.PY - Mixed Relative/Absolute Paths in Modal Context
**File:** `src/modal_train.py`  
**Lines:** 39-41, 265-286  
**Severity:** HIGH (can cause upload failures)

**Problem:**
- Lines 39-41 define container paths (correct - absolute):
  ```python
  DATASET_PATH = Path("/dataset")
  CACHE_PATH   = Path("/hf-cache")
  OUTPUT_PATH  = Path("/output")
  ```

- Lines 265-286 use local (host) relative paths:
  ```python
  local_json = Path("data") / dataset_filename
  local_json = Path("data/synthetic_factory/synthetic_dataset.json")
  local_screenshots = Path("data/ui_screenshots")
  local_synthetic_screenshots = Path("data/synthetic_factory/screenshots")
  ```

**Risk:**
- If script runs from non-root directory, upload will fail with:
  - `FileNotFoundError: [Errno 2] No such file or directory: 'data/ui_screenshots'`
  
**Documented Workaround:**
- Comment on line 260 states: "Run once (or when dataset changes)"
- Implies script is meant to run from project root only
- ✓ This is acceptable but should be in docstring of `upload_dataset()`

**Recommendation:**
Add to upload_dataset() docstring:
```python
def upload_dataset(dataset_filename: str = "dataset_vision.json"):
    """
    Must be run from project root directory.
    Example: cd /home/harshil/ghost_architect_gemma3 && modal run src/modal_train.py::upload_dataset
    """
```

---

### Issue #4: APP.PY - Hardcoded Default Adapter Path Mismatch
**File:** `src/app.py`  
**Line:** 37  
**Severity:** MEDIUM (UX issue)

```python
value="output/adapters/trinity_a10g"
```

**Problem:**
- Default path `trinity_a10g` created by Modal training script
- Alternative path `phase2` created by Colab training
- Comment on line 40 acknowledges this: "Change this to output/adapters/phase2 if you trained on Colab"
- Confuses users about which adapter to use

**Current Paths in System:**
```
output/adapters/
├── phase1/       (old - from Colab)
├── phase2/       (Colab training)
└── trinity_a10g/ (Modal training)
```

**Recommendation:**
Detect which adapter exists:
```python
default_adapter = "output/adapters/trinity_a10g"
if not Path(default_adapter).exists() and Path("output/adapters/phase2").exists():
    default_adapter = "output/adapters/phase2"

adapter_dir = st.text_input(
    "Adapter path",
    value=default_adapter,
    help="Path to your trained LoRA adapter directory",
)
```

---

## MEDIUM PRIORITY ISSUES

### Issue #5: EXPORT.PY - Mismatched Default Adapter Path
**File:** `src/export.py`  
**Line:** 81  
**Severity:** MEDIUM (documentation issue)

```python
parser.add_argument("--adapter_dir", type=str, default="output/adapters/phase2", ...)
```

**Problem:**
- Default is `phase2` (for Colab) but modal training creates `trinity_a10g`
- Users must specify `--adapter_dir output/adapters/trinity_a10g` after Modal training
- ✓ This is documented but could be clearer

---

### Issue #6: TRAIN_VISION.PY - Hardcoded Path Dependency
**File:** `src/train_vision.py`  
**Line:** 169  
**Severity:** MEDIUM (UX friction)

```python
parser.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
```

**Problem:**
- Must be run from project root (relative path)
- No check if path exists before attempting to load
- Fails with unclear error if run from wrong directory

**Example Failure:**
```bash
$ cd src && python train_vision.py
FileNotFoundError: [Errno 2] No such file or directory: 'data/dataset.json'
```

**Recommendation:**
Add validation:
```python
def main(args):
    if not args.dataset.exists():
        print(f"ERROR: Dataset not found at {args.dataset}")
        print(f"Make sure you're in the project root directory")
        sys.exit(1)
```

---

### Issue #7: DATA_FACTORY.PY - Partial Model Availability Coverage
**File:** `scripts/data_factory.py`  
**Lines:** 49-58  
**Severity:** LOW-MEDIUM (gracefully handled)

**Models listed:**
```python
'gemini-2.0-flash'              # ✓ Stable
'gemini-2.5-flash'              # ⚠️ Newer
'gemini-flash-lite-latest'       # ✓ Latest alias (safe)
'gemini-flash-latest'            # ✓ Latest alias (safe)
'gemini-pro-latest'              # ✓ Latest alias (safe)
'gemini-3.1-flash-lite-preview'  # ⚠️ Preview (may not exist)
'gemini-2.0-flash-lite-001'      # ✓ Specific version
'gemini-2.5-pro'                 # ⚠️ New
```

**Problem:**
- Some models may be unavailable or in preview
- Code handles this gracefully with 429 error and model rotation
- ✓ **No action needed** - working as intended

---

### Issue #8: BUILD_VISION_DATASET.PY - Silent File Skip on Corruption
**File:** `scripts/build_vision_dataset.py`  
**Line:** 189  
**Severity:** LOW (correctly handled)

```python
if not validate_image(img_path):
    invalid_count += 1
    continue
```

**Observation:**
- Silently skips corrupted images without detailed logging
- User only sees count in summary
- ✓ This is acceptable - error printed via validate_image()

---

## ENVIRONMENT VARIABLE ISSUES

### Incomplete Environment Configuration
**Severity:** MEDIUM (causes runtime failures)

**Missing from .env:**
```
HF_HOME              (used in modal_train.py:163)
TRANSFORMERS_CACHE   (used in modal_train.py:164)
HF_DATASETS_CACHE    (used in modal_train.py:165)
```

**Present in .env:**
```
✓ HUGGINGFACE_TOKEN / HF_TOKEN
✓ GEMINI_API_KEY
```

**Impact:**
- Modal training will fail if HF_TOKEN not set
- Local training needs HuggingFace auth
- Cache directories are set but values may not match .env.example

**Recommendation:**
Update `.env.example` to include Modal-specific vars:
```bash
HF_TOKEN=hf_xxxxxxxxxxxx
GEMINI_API_KEY=xxxxxxxxxx
HF_HOME=/hf-cache          # For Modal volumes
TRANSFORMERS_CACHE=/hf-cache/transformers
HF_DATASETS_CACHE=/hf-cache/datasets
```

---

## IMPORT & DEPENDENCY ISSUES

### All imports verified to exist
**Status:** ✓ PASS

**Core dependencies:**
- ✓ `torch`, `transformers` - Main ML stack
- ✓ `unsloth` - Vision model support
- ✓ `peft` - LoRA adapters
- ✓ `datasets` - HF datasets
- ✓ `streamlit` - Web UI
- ✓ `rich` - CLI formatting
- ✓ `PIL` - Image handling
- ✓ `playwright` - Screenshot capture
- ⚠️ `modal` - Only needed for Modal deployment

---

## DATASET & PATH CONSISTENCY

### Dataset Schema Validation
**Status:** ✓ PASS

**Files checked:**
```
✓ data/dataset_vision.json   (287 items)
✓ data/dataset_merged.json   (5287 items)
✓ data/ui_screenshots/       (directory exists)
✓ data/synthetic_factory/    (directory exists)
```

**Field consistency:**
- ✓ All items have `image_path` field (never `image` in JSON)
- ✓ All items have `instruction` field
- ✓ All items have `output` field
- ✓ Code correctly references `item.get("image_path")`

**Generated paths in memory:**
- modal_train.py:114: `DATASET_PATH / "ui_screenshots" / filename` ✓ Correct
- train_vision.py:47: `Path(item.get("image_path"))` ✓ Correct
- app.py:221: `str(tmp_path)` ✓ Uses temp file ✓ Correct

---

## FUNCTION SIGNATURES & CALLS

### Critical Functions Verified
**Status:** ✓ PASS (with noted redundancy)

1. **`_load_dataset()`** (modal_train.py:98)
   - ✓ Returns single Dataset object
   - ✓ Called correctly: `dataset = _load_dataset(dataset_json)`

2. **`load_vision_dataset()`** (train_vision.py:37)
   - ✓ Returns single Dataset object  
   - ✓ Called correctly: `dataset = load_vision_dataset(dataset_path)`

3. **`create_vision_model()`** (train_vision.py:85)
   - ✓ Returns tuple `(model, processor)`
   - ✓ Called correctly: `model, processor = create_vision_model()`

4. **`_render_rich()`** (inference.py:62)
   - ⚠️ Has unused `raw_sql` parameter
   - ⚠️ Called with: `_render_rich(img_file.name, sql, sql)` (redundant)
   - **Should fix:** Remove unused parameter

5. **`FastVisionModel.from_pretrained()`**
   - ✓ Called correctly in: app.py, inference.py, export.py, modal_train.py, train_vision.py
   - ✓ Parameters: `model_name`, `load_in_4bit=True`

6. **`SFTTrainer` configuration**
   - ✓ Consistent across modal_train.py and train_vision.py
   - ✓ Both set: `dataset_text_field=""`, `skip_prepare_dataset=True`
   - ✓ Both use: `UnslothVisionDataCollator`

---

## CONNECTION ISSUES BETWEEN FILES

### File Dependencies
**Status:** ✓ MOSTLY CORRECT

**Training Pipeline:**
```
data/dataset_vision.json
  ↓ (read by)
train_vision.py OR modal_train.py
  ↓ (creates)
output/adapters/{phase2 OR trinity_a10g}/
  ↓ (read by)
export.py, app.py, inference.py
```

**Synthetic Data Pipeline:**
```
scripts/data_factory.py
  ↓ (creates)
data/synthetic_factory/synthetic_dataset.json
  ↓ (read by)
scripts/merge_datasets.py
  ↓ (creates)
data/dataset_merged.json
  ↓ (can be used by)
train_vision.py --dataset data/dataset_merged.json
```

**Cross-file Message Format:**
- ✓ All files use consistent message format: `[{role, content: [{type, image, text}]}]`
- ✓ Consistent instruction text: "Analyze the UI structure and generate an appropriate database schema."

---

## HARDCODED PATHS AUDIT

### Relative Paths (require project root execution)
```
scripts/merge_datasets.py:15         data/synthetic_factory/synthetic_dataset.json
scripts/build_vision_dataset.py:165  data/ui_screenshots
src/inference.py:144                 data/ui_screenshots/*.png
src/app.py:37                        output/adapters/trinity_a10g
```

**Status:** ✓ Acceptable - all scripts must run from project root

### Absolute Modal Paths (container-specific)
```
modal_train.py:39    /dataset
modal_train.py:40    /hf-cache
modal_train.py:41    /output
```

**Status:** ✓ Correct - these are inside the Modal container

---

## SUMMARY TABLE

| File | Type | Issue | Line | Severity |
|------|------|-------|------|----------|
| train_vision.py | Bug | Wrong default dataset path | 169 | **CRITICAL** |
| inference.py | Code Quality | Unused function parameter | 62, 189 | Medium |
| modal_train.py | Documentation | Missing path execution context | 265 | Medium |
| app.py | UX | Mismatched default adapter path | 37 | Medium |
| export.py | Documentation | Different default than Modal | 81 | Low |
| data_factory.py | Info | Some models may be preview | 49-58 | Low |
| merge_datasets.py | Info | Requires project root execution | 15-17 | Low |
| build_vision_dataset.py | Info | Silent image skip | 189 | Low |

---

## RECOMMENDATIONS (Priority Order)

### IMMEDIATE (Do first)
1. **Fix train_vision.py default dataset path** → line 169
   - Change from `data/dataset.json` to `data/dataset_vision.json`
   - **Time:** 1 minute

2. **Remove unused raw_sql parameter** → inference.py lines 62, 189
   - Delete from function signature and call
   - **Time:** 2 minutes

### SHORT-TERM (Before production)
3. **Add path validation to train_vision.py**
   - Check dataset exists with helpful error message
   - **Time:** 5 minutes

4. **Add adapter auto-detection to app.py**
   - Detect which adapter exists and set default
   - **Time:** 10 minutes

5. **Document environment variables**
   - Update .env.example with all required vars
   - **Time:** 5 minutes

### NICE-TO-HAVE (Polish)
6. Standardize default adapter names across scripts
7. Add working directory validation to scripts
8. Enhance logging for skipped files in build_vision_dataset.py

---

## TESTING RECOMMENDATIONS

### Smoke Tests
```bash
# Test each script can be imported without error
python3 -c "from src.modal_train import train"
python3 -c "from src.train_vision import train_vision_model"
python3 -c "from src.app import parse_create_tables"
python3 -c "from src.inference import run_inference"
python3 -c "from src.export import export_to_gguf"
python3 -c "from scripts.build_vision_dataset import build_vision_dataset"
python3 -c "from scripts.merge_datasets import merge_datasets"
python3 -c "from scripts.data_factory import main"
```

### Integration Tests
```bash
# Test dataset loading works
python src/train_vision.py --dataset data/dataset_vision.json --output /tmp/test_train

# Test inference
python src/inference.py --adapter output/adapters/phase2 --image data/ui_screenshots/sample.png

# Test export
python src/export.py --adapter_dir output/adapters/phase2 --output_dir /tmp/gguf_test
```

---

**Audit Completed:** 2025-03-10  
**Auditor:** Code Audit Agent  
**Status:** Ready for deployment with recommended fixes
