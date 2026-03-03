# Dataset Setup Guide for Ghost Architect

## Current Status ✅

| Component | Status | Notes |
|-----------|--------|-------|
| **data/dataset.json** | ✅ Ready | 30 starter examples for Phase 1 text training |
| **data/dataset_vision.json** | ✅ Ready | 287 vision training examples (UI screenshot → SQL) |
| **scripts/download_datasets.py** | ✅ Fixed | 5 security fixes, URL validation improved |
| **data/ui_screenshots/** | ✅ Ready | 287 PNGs (107MB) |

---

## What Just Happened

### 1. Fixed `scripts/download_datasets.py` (5 security patches)
- ✅ Removed duplicate docstring
- ✅ Added URL validation (rejects malformed URLs, tracking params, localhost)
- ✅ Fixed filename collision risk with hash-based naming
- ✅ Replaced silent exception handling with typed error logging
- ✅ Reduced timeout from 15s → 10s (fail-fast behavior)

**Run it again to expand screenshots:**
```bash
# Change MAX_URLS in scripts/download_datasets.py to increase (currently 50)
uv run python scripts/download_datasets.py
```

### 2. Created `scripts/generate_training_data.py`
This generates **30 starter code examples** for Phase 1 training. Format:
```json
[
  {
    "instruction": "Write a Python function to sum two numbers.",
    "input": "a=5, b=10",
    "output": "def sum_numbers(a, b):\n    return a + b\n..."
  }
]
```

**Already generated:** `data/dataset.json` ✅

---

## Phase 1 Training Next Steps

### Option A: Use Starter Data (Now)
**Already done!** You have `data/dataset.json` with 30 examples ready.

Run Phase 1 training on Colab:
```python
# In Colab notebook
!cd /content/ghost_architect_gemma3 && python src/train.py \
  --config configs/training_config.yaml \
  --dataset data/dataset.json
```

### Option B: Expand Starter Data (Recommended)
Run this script to create **300+ examples**:
```bash
# Edit scripts/generate_training_data.py to add more domains/problem categories
# Then regenerate:
uv run python scripts/generate_training_data.py
```

### Option C: Import from HuggingFace (Production)
Replace `data/dataset.json` with real-world data:

```python
from datasets import load_dataset
import json

# Load Alpaca-style dataset (code-focused)
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
examples = dataset['train'][:300]  # Take first 300

# Convert to our format
data = [
    {
        "instruction": ex['instruction'],
        "input": ex['input'],
        "output": ex['output']
    }
    for ex in examples
]

with open('data/dataset.json', 'w') as f:
    json.dump(data, f)
```

---

## Dataset Format Specification

Your `data/dataset.json` must follow this format:

```json
[
  {
    "instruction": "What to do (required)",
    "input": "Optional context or function params",
    "output": "The expected result/code"
  }
]
```

**Constraints:**
- ✅ Min 10 examples (Phase 1 can work with this)
- ✅ Recommended 300+ examples (better convergence)
- ✅ Max ~50K examples (T4 memory limit)
- ✅ `instruction` field required; `input` can be empty string `""`
- ✅ No special characters in JSON—use `\n` for newlines

---

## UI Screenshots (Phase 2)

**Current:** 287 screenshots in `data/ui_screenshots/` (107MB), paired with SQL annotations in `data/dataset_vision.json`.

**To expand:**
```bash
# Edit MAX_URLS in scripts/download_datasets.py
# max_urls = 1000  # or higher
uv run python scripts/download_datasets.py
# Then rebuild vision dataset:
uv run python scripts/build_vision_dataset.py
```

**How they're used:**
- Phase 2 vision training via `src/train_vision.py` (Colab T4) or `src/modal_train.py` (Modal A10G)
- Not needed for Phase 1 text fine-tuning

---

## Troubleshooting

### Issue: `Dataset file is empty: data/dataset.json`
**Solution:** Run `uv run python scripts/generate_training_data.py`

### Issue: Too many failed URLs in screenshot capture
**Root cause:** CSV data contains redirect links (ProductHunt affiliate URLs)
**Solution:** Already fixed in updated `scripts/download_datasets.py`; filters out `utm_` params

### Issue: Want different coding domain (e.g., JavaScript, SQL)
**Solution:**
1. Edit `scripts/generate_training_data.py`
2. Replace Python examples with domain-specific code
3. Run: `uv run python scripts/generate_training_data.py`

---

## File Layout

```
ghost_architect_gemma3/
├── data/
│   ├── dataset.json                    # Phase 1 text training data (30 examples)
│   ├── dataset_vision.json             # Phase 2 vision training data (287 examples)
│   ├── raw_csvs/                       # Source CSVs for scraper
│   │   ├── saas_companies.csv
│   │   ├── ycombinator.csv
│   │   └── producthunt.csv
│   ├── ui_screenshots/                 # 287 PNGs (Playwright captures)
│   │   ├── producthunt.com_12345.png
│   │   └── ...
│   ├── synthetic_pairs/                # (empty, for future use)
│   └── validation_set/                 # (empty, for future use)
│
└── scripts/
    ├── build_vision_dataset.py          # Builds dataset_vision.json from screenshots + Gemini
    ├── download_datasets.py             # Playwright scraper for UI screenshots
    ├── generate_training_data.py        # Generates Phase 1 starter data
    ├── validate_dataset.py              # Validates dataset.json (make dataset-check)
    └── validate_environment.py          # Validates GPU/deps (make validate)
```

---

## Next Actions

1. **[Now]** Verify datasets are ready:
   ```bash
   make dataset-check                     # validates data/dataset.json
   python scripts/build_vision_dataset.py  # validates dataset_vision.json
   ```

2. **[Colab]** Run Phase 1 training:
   ```python
   # In notebook, run cell 6 to launch training
   !python src/train.py --config configs/training_config.yaml --dataset data/dataset.json
   ```

3. **[Vision training]** Choose a training path:
   ```bash
   # Modal A10G (full Trinity — QLoRA+DoRA+rsLoRA):
   modal run src/modal_train.py
   # OR Colab T4 (QLoRA+rsLoRA, no DoRA):
   python src/train_vision.py
   ```

4. **[After training]** Export model to GGUF:
   ```bash
   python src/export.py
   ```

---

## Learning Resources

- **LoRA/QLoRA fundamentals:** `docs/learning-guide.md`
- **Trinity architecture:** `docs/architecture.md`
- **Memory budget walkthrough:** `docs/plan.md` (Section 1.3)
- **OOM recovery protocol:** `README.md` (Risk Management)

