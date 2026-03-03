# Data Factory Setup Guide

## Overview
The Data Factory generates **5,000+ synthetic UI/SQL pairs** from scratch using:
1. **Gemini API** - Generate UI descriptions
2. **Jinja2** - Render HTML from descriptions  
3. **Playwright** - Screenshot HTML pages
4. **Gemini Vision API** - Generate SQL schemas from screenshots

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
playwright install  # Install browser drivers
```

### 2. Get Gemini API Key
1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key
4. Create `.env` file (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```
5. Edit `.env` and add:
   ```
   GEMINI_API_KEY=your_key_here
   ```

**⚠️ Cost:** The free tier Gemini API is:
- **Rate limit:** 15 requests/minute
- **Daily quota:** 1M tokens/day
- **Sufficient for:** ~5,000 images (takes ~8-10 hours)

## Quick Start

### Generate 100 UI/SQL Pairs (Testing)
```bash
python scripts/data_factory.py 100
```

This will:
1. Generate 100 UI descriptions using Gemini
2. Render them to HTML using Tailwind CSS
3. Screenshot each HTML page
4. Generate SQL schema for each screenshot

**Output:** `data/synthetic_factory/`
- `ui_descriptions.json` - UI metadata
- `html_pages/` - Rendered HTML files
- `screenshots/` - PNG screenshots (1280x720)
- `synthetic_dataset.json` - Final UI/SQL pairs

**Time estimate:** ~20-30 minutes (respecting rate limits)

### Generate 5,000 UI/SQL Pairs (Full Scale)
```bash
python scripts/data_factory.py 5000
```

**Time estimate:** ~8-10 hours  
**Storage:** ~15-20 GB (5,000 screenshots × 300-400 KB each)

## Output Format

The `synthetic_dataset.json` will look like:
```json
[
  {
    "image_path": "data/synthetic_factory/screenshots/ui_0000.png",
    "instruction": "Analyze this UI and generate the PostgreSQL schema.",
    "output": "CREATE TABLE users (...); CREATE TABLE orders (...);",
    "domain": "synthetic",
    "size_kb": 285.4
  },
  ...
]
```

This matches the format expected by `src/synthetic_generator.py` and the training pipeline.

## Integration with Training

Once you have synthetic data, merge it with existing data:

```bash
# Merge synthetic data with existing dataset_vision.json
python scripts/merge_datasets.py \
  --synthetic data/synthetic_factory/synthetic_dataset.json \
  --existing data/dataset_vision.json \
  --output data/merged_dataset.json
```

Then train with merged dataset:
```bash
python src/train_vision.py --dataset data/merged_dataset.json
```

## Monitoring & Resume

- The factory **saves progress incrementally**
- If interrupted, re-running the same command **resumes from where it left off**
- Check `data/synthetic_factory/` for intermediate results

## Troubleshooting

### "GEMINI_API_KEY not found"
- Verify `.env` file exists and has correct key
- Run: `echo $GEMINI_API_KEY` to check environment

### "Rate limit exceeded"
- Free tier has 15 req/min limit
- Script automatically waits between requests
- Consider using a paid tier for faster generation

### "Playwright browser not found"
- Run: `playwright install`
- This downloads Chrome/Firefox/WebKit

### "jinja2 not found"
- Run: `pip install jinja2`

## Advanced Options

### Custom UI Types
Edit `scripts/data_factory.py` in the `ui_types` list to generate specialized UIs:
```python
ui_types = [
    "Medical Records Dashboard",
    "Flight Booking System",
    "Real Estate Listing Portal",
    # ... add your own
]
```

### Custom HTML Template
Modify the `HTML_TEMPLATE` in `scripts/data_factory.py` to generate different visual styles.

## Next Steps

After generating synthetic data:
1. **Validate** SQL schemas (check for syntax errors)
2. **Merge** with existing dataset
3. **Train** Gemma-3 using fine-tuning pipeline
4. **Evaluate** on held-out test set

---

**Cost Summary:**
- **Gemini API:** Free tier sufficient (1M tokens/day)
- **Playwright:** Free (open-source)
- **Jinja2:** Free (open-source)
- **Total:** $0 for 5,000 synthetic UI/SQL pairs ✅
