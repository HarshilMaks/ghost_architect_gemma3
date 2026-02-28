# Phase 2: Vision Training for Ghost Architect

## Overview

Train Gemma-3-12B on 287 UI screenshots to create a UI-to-SQL model. Two training paths are available:

- **Modal A10G** (`src/modal_train.py`): Full Trinity (QLoRA + DoRA + rsLoRA), `finetune_vision_layers=True`, 3 epochs, 4096 ctx, ~$1.65
- **Colab T4** (`src/train_vision.py`): QLoRA + rsLoRA only (DoRA disabled due to dtype bug), `finetune_vision_layers=False`, 1 epoch, 2048 ctx, free

**What you're building:**
- 🖼️  Vision model that understands UI layouts
- 🧠 SQL schema generation from visual input
- 📦 LoRA adapters exportable to GGUF
- 🚀 Ollama deployment + Streamlit demo app

---

## Dataset: 287 UI Screenshots

### Status
✅ **dataset_vision.json** created and validated
- 287 unique UI screenshots (107MB total)
- Classified by domain (dashboards, e-commerce, admin panels, etc.)
- Annotated with synthetic SQL schemas via Gemini API

### Directory
```
data/
├── ui_screenshots/           # Original images (287 PNG files)
│   ├── hangman-c0h7.onrender.com_7455.png
│   ├── dashboard-de-performance-de-tr-fego.vercel.app_6896.png
│   └── ... (285 more)
└── dataset_vision.json       # Training annotations (287 examples)
```

### Dataset Format
Image paths are embedded in messages (no top-level `images` column):
```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "data/ui_screenshots/example.png", "text": ""},
      {"type": "text", "text": "Analyze this UI and generate the database schema."}
    ]},
    {"role": "assistant", "content": "CREATE TABLE products (...)"}
  ]
}
```
This avoids TRL's `_is_vision_dataset` check. `UnslothVisionDataCollator` falls back to `process_vision_info()` → `fetch_image()` → `Image.open(path)`.

---

## Training Architecture: Two Paths

### Path A: Modal A10G (`src/modal_train.py`) — Full Trinity
- **QLoRA + DoRA + rsLoRA** (DoRA dtype bug monkey-patched inline)
- `finetune_vision_layers=True`
- 3 epochs, 4096 context, ~$1.65 from free $30 Modal credits
- Best quality results

### Path B: Colab T4 (`src/train_vision.py`) — QLoRA + rsLoRA
- **No DoRA** (PEFT's dora.py has fp16/fp32 dtype mismatch with Unsloth's Gemma3 patch)
- `finetune_vision_layers=False`
- 1 epoch, 2048 context, free
- Good baseline results

### Memory Budget (T4: 16GB)
```
Model weights (4-bit QLoRA):     7.6 GB
Gradients (rank 64 + DoRA):      5.5 GB
Vision embeddings (4096 seq):    2.5 GB
Overhead & buffers:              ~0.4 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                          ~15.6 GB (98% utilization)
```

---

## Step-by-Step Training Guide

### Prerequisites
- ✅ HuggingFace token in `.env`
- ✅ HUGGINGFACE_TOKEN authenticated
- ✅ Google Colab with T4 GPU or local NVIDIA GPU

### 1️⃣ Verify Dataset (Local or Colab)

```bash
# Check dataset integrity
python3 scripts/build_vision_dataset.py

# Expected output:
# ✅ Validation complete!
#    • Valid images: 287
#    • Duplicates removed: 0
```

### 2️⃣ Start Training

**Option A: Modal A10G (recommended)**
```bash
modal run src/modal_train.py
```

**Option B: Colab T4 (free)**

In Colab notebook:
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/ghost_architect_gemma3

!pip install -q unsloth "xformers<0.0.27" trl peft transformers datasets
!pip install -q python-dotenv huggingface-hub pillow

!python src/train_vision.py
```

### 3️⃣ Monitor Training (Live VRAM)

**In separate Colab cell:**
```bash
!watch -n 2 nvidia-smi
```

**Expected GPU metrics:**
- GPU Util: 85-95%
- VRAM: 14.8-15.6 GB (showing red is OK - we want 98% utilization)
- Temperature: <75°C

### 4️⃣ Training Time Estimate

| Dataset | T4 GPU | A10G GPU |
|---------|--------|----------|
| 287 examples (full) | 60-120 min | 15-25 min |
| 50 examples (test) | 20-30 min | 3-5 min |

---

## What Happens During Training

### Phase 1: Model Loading (5-10 min)
```
✅ HuggingFace authenticated
Loading Gemma-3 vision model...
✓ Downloaded model shards (25GB → 7.6GB after quantization)
✅ Model loaded with Trinity adapters
   Trainable params: 1,247,232
```

### Phase 2: Training Loop (40-80 min)
```
Dataset splits: 287 examples

Step 5/287    loss: 2.14
Step 10/287   loss: 1.92
Step 20/287   loss: 1.54
Step 50/287   loss: 0.87
Step 100/287  loss: 0.42
Step 287/287  loss: 0.18  ✅
```

### Phase 3: Saving (2-5 min)
```
✅ Training complete! Saving to output/adapters/vision_trinity...
✨ NEXT: Export to GGUF for production inference
```

---

## Output Files

After training, you'll have:

```
output/adapters/vision_trinity/
├── adapter_config.json          # LoRA configuration
├── adapter_model.bin            # LoRA weights (lightweight)
├── preprocessor_config.json     # Image processor settings
├── config.json                  # Model config
└── generation_config.json       # Generation parameters

logs/vision/
└── events.out.tfevents.*        # Training metrics
```

**Size**: LoRA weights ~8-15 MB (vs 24 GB for full model!)

---

## Troubleshooting

### ❌ "CUDA Out of Memory" Error

**Fix 1: Reduce batch size** (already minimal at 1)
```python
# In train_vision.py or modal_train.py:
per_device_train_batch_size=1    # Already set
gradient_accumulation_steps=2    # Reduce from 4 to 2
```

**Fix 2: Lower learning rank**
```python
r=32,  # Reduce from 64 to 32
```

**Fix 3: Use fewer examples**
```bash
# Edit the training script to limit dataset size
```

### ❌ "HUGGINGFACE_TOKEN not found"

**Solution**: Add to `.env` file
```bash
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

Then restart Colab runtime.

### ❌ "Access to model google/gemma-3-12b-it is restricted"

**Solution**: Visit https://huggingface.co/google/gemma-3-12b-it
1. Click "Access Repository"
2. Accept Google's license terms
3. Wait ~1 minute for approval
4. Retry training

---

## Next Steps After Training

### Export to GGUF

```bash
# Convert LoRA adapters to GGUF for Ollama
python src/export.py
```

### Deploy with Ollama

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./gemma3-vision-ui2sql.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are an expert in UI analysis and SQL schema design."
EOF

# Build and run
ollama create gemma3-vision-ui2sql -f Modelfile
ollama run gemma3-vision-ui2sql "Analyze this UI..."
```

### Use in Streamlit Demo

```bash
streamlit run src/app.py
# Upload a screenshot → see the generated schema
```

### CLI Testing

```bash
python src/inference.py
# Rich terminal output for quick testing
```

---

## Production Checklist

- [ ] Dataset built (287 examples): `data/dataset_vision.json`
- [ ] HuggingFace authenticated: `.env` has `HUGGINGFACE_TOKEN`
- [ ] Training completed: `output/adapters/` (via Modal or Colab)
- [ ] GGUF exported: `python src/export.py`
- [ ] Ollama tested: `ollama run ghost-architect`
- [ ] Streamlit demo: `streamlit run src/app.py`

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/build_vision_dataset.py` | Create dataset from UI screenshots + Gemini |
| `src/modal_train.py` | Modal A10G training (full Trinity) |
| `src/train_vision.py` | Colab T4 training (QLoRA+rsLoRA, no DoRA) |
| `src/export.py` | GGUF export for Ollama |
| `src/app.py` | Streamlit demo app |
| `src/inference.py` | CLI testing with rich output |
| `data/dataset_vision.json` | Training annotations (287 examples) |
| `docs/phase1_trinity_training.md` | Phase 1 text training (reference) |

---

## Learning Resources

**Understanding Trinity Architecture:**
- QLoRA: https://arxiv.org/pdf/2305.14314
- DoRA: https://arxiv.org/pdf/2402.09353
- Rank-Stabilized LoRA: Built into Unsloth

**Gemma-3 Vision:**
- Model Card: https://huggingface.co/google/gemma-3-12b-it
- Technical Paper: https://arxiv.org/pdf/2010.11929

**Vision Training Best Practices:**
- Image preprocessing for vision transformers
- Multi-modal tokenization strategies
- Handling variable-size images

---

**Status**: ✅ Ready for training

**Next**: Run `modal run src/modal_train.py` (Modal A10G) or `python src/train_vision.py` (Colab T4)

