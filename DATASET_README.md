# Dataset Setup Guide for Ghost Architect Phase 1

## Current Status ✅

| Component | Status | Notes |
|-----------|--------|-------|
| **data/dataset.json** | ✅ Ready | 30 starter examples for Phase 1 training |
| **scripts/download_datasets.py** | ✅ Fixed | 5 security fixes, URL validation improved |
| **data/ui_screenshots/** | 🔄 Ongoing | ~50+ UI images captured, can expand to 10K+ |

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
  --config configs/training_config_colab_t4.yaml \
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

## UI Screenshots (Phase 2 Preparation)

**Current:** ~50 screenshots captured in `data/ui_screenshots/`

**To expand:**
```bash
# Edit MAX_URLS in scripts/download_datasets.py
# max_urls = 1000  # or higher
uv run python scripts/download_datasets.py
```

**How they're used:**
- Phase 2: Vision encoder + text → SQL generation
- Not needed for Phase 1 text fine-tuning
- Good to gather in parallel while Phase 1 trains

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
│   ├── dataset.json                    # Training data (auto-generated)
│   ├── raw_csvs/                       # Your input CSVs
│   │   ├── saas_companies.csv
│   │   ├── ycombinator.csv
│   │   └── producthunt.csv
│   └── ui_screenshots/                 # Playwright captures (Phase 2)
│       ├── producthunt.com_12345.png
│       └── ... (more screenshots)
│
└── scripts/
    ├── download_datasets.py             # Screenshot extractor (FIXED)
    └── generate_training_data.py        # Data generator (NEW)
```

---

## Next Phase 1 Actions

1. **[Now]** Verify `data/dataset.json` is ready:
   ```bash
   wc -l data/dataset.json
   head -20 data/dataset.json
   ```

2. **[Colab]** Run Phase 1 training:
   ```python
   # In notebook, run cell 6 to launch training
   !python src/train.py --config configs/training_config_colab_t4.yaml --dataset data/dataset.json
   ```

3. **[After training]** Export model to GGUF:
   ```bash
   # src/export.py (to be implemented)
   python src/export.py --adapter output/adapters/final_adapter
   ```

---

## Learning Resources

- **LoRA/QLoRA fundamentals:** `docs/learning-guide.md`
- **Trinity architecture:** `docs/architecture.md`
- **Memory budget walkthrough:** `docs/plan.md` (Section 1.3)
- **OOM recovery protocol:** `README.md` (Risk Management)

