#!/usr/bin/env python3
"""
Ghost Architect — Modal Serverless Training
Full Trinity stack: QLoRA + DoRA + rsLoRA on A10G (24GB VRAM).

DoRA works on A10G because:
  - A10G supports bfloat16 → Unsloth doesn't need its fp16 Gemma3 attention hack
  - We also include the DoRA dtype patch as a safety net

SETUP (one time):
  pip install modal
  modal setup                          # creates ~/.modal.toml, links to your account
  modal secret create ghost-architect-secrets HF_TOKEN=hf_xxxx

UPLOAD DATASET (one time, or when dataset changes):
  modal run src/modal_train.py::upload_dataset

RUN TRAINING (~1.5 hrs on A10G, ~$1.65 from your $30 credits):
  modal run src/modal_train.py

DOWNLOAD ADAPTER (after training):
  modal run src/modal_train.py::download_adapter
"""

import modal
from pathlib import Path

# ── Persistent Volumes ──────────────────────────────────────────────────────
# Volumes persist across Modal runs.
# model_cache_vol saves the 12GB Gemma weights — avoids re-downloading every run.
# dataset_vol holds your 287 screenshots + dataset_vision.json.
# output_vol holds the trained LoRA adapter.

dataset_vol     = modal.Volume.from_name("ghost-architect-dataset",    create_if_missing=True)
model_cache_vol = modal.Volume.from_name("ghost-architect-hf-cache",   create_if_missing=True)
output_vol      = modal.Volume.from_name("ghost-architect-output",     create_if_missing=True)

# Paths INSIDE the container (where volumes are mounted)
DATASET_PATH = Path("/dataset")
CACHE_PATH   = Path("/hf-cache")
OUTPUT_PATH  = Path("/output")

# ── Container Image ─────────────────────────────────────────────────────────
# Modal builds this once and caches it. Only rebuilds when pip_install changes.
# Using CUDA 12.1 base which matches Unsloth's requirements.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "curl")
    .pip_install(
        # Core ML stack — pin torch first so Unsloth picks the right CUDA variant
        "torch==2.4.1",
        "torchvision==0.19.1",
        # Unsloth from source — gets latest Gemma3 patches
        "unsloth @ git+https://github.com/unslothai/unsloth.git",
        "unsloth_zoo",
        # Dependencies
        "transformers>=4.46.0",
        "trl>=0.11.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.44.0",
        "xformers",
        "datasets>=3.0.0",
        "huggingface_hub>=0.25.0",
        "pillow>=10.0.0",
        "triton",
        "sentencepiece",
    )
)

app = modal.App("ghost-architect-training", image=image)


# ── DoRA Dtype Patch ─────────────────────────────────────────────────────────
# PEFT's dora.py passes x_eye in the input's dtype to lora_A (fp32) without
# casting first. Regular LoRA does `x = x.to(self.lora_A.weight.dtype)` first.
# This patch adds the missing cast. Applied at runtime inside the container.
def _patch_dora():
    import peft.tuners.lora.dora as _dora_mod
    _dora_path = _dora_mod.__file__
    with open(_dora_path, "r") as f:
        src = f.read()
    old = "        lora_weight = lora_B(lora_A(x_eye)).T"
    new = (
        "        x_eye = x_eye.to(next(lora_A.parameters()).dtype)"
        "  # cast to match lora_A weights\n"
        "        lora_weight = lora_B(lora_A(x_eye)).T"
    )
    if old in src:
        with open(_dora_path, "w") as f:
            f.write(src.replace(old, new, 1))
        print("✅ DoRA dtype patch applied")
    else:
        print("ℹ️  DoRA patch: already applied or PEFT version changed — skipping")


# ── Dataset Loader ────────────────────────────────────────────────────────────
def _load_dataset(dataset_json: Path):
    import json
    import logging
    from datasets import Dataset

    log = logging.getLogger(__name__)
    parsed = json.loads(dataset_json.read_text())
    valid, skipped = [], 0

    for item in parsed:
        # image_path in JSON was written as data/ui_screenshots/xxx.png (relative)
        # Inside the container it lives at /dataset/ui_screenshots/xxx.png
        raw_path = item.get("image_path") or item.get("image", "")
        # Strip any leading "data/" prefix so it resolves relative to DATASET_PATH
        rel = raw_path.replace("data/ui_screenshots/", "")
        img_path = DATASET_PATH / "ui_screenshots" / rel

        if not img_path.exists():
            skipped += 1
            continue

        # Image path string embedded in messages — no top-level 'images' column.
        # UnslothVisionDataCollator falls back to process_vision_info(messages)
        # which calls fetch_image() → Image.open(path) for local paths.
        # Consistent Arrow schema: all content items have {type, image, text}.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path), "text": ""},
                    {"type": "text",  "image": "",            "text": item["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "image": "", "text": item["output"]},
                ],
            },
        ]
        valid.append({"messages": messages})

    log.info(f"✅ Dataset: {len(valid)} valid, {skipped} skipped")
    return Dataset.from_list(valid)


# ── Main Training Function ────────────────────────────────────────────────────
@app.function(
    gpu="A10G",          # 24GB VRAM — enough for full Trinity including DoRA
    timeout=7200,        # 2 hour cap (training ~1.5 hrs)
    volumes={
        DATASET_PATH: dataset_vol,
        CACHE_PATH:   model_cache_vol,
        OUTPUT_PATH:  output_vol,
    },
    secrets=[modal.Secret.from_name("ghost-architect-secrets")],
)
def train():
    import os
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Point HuggingFace cache at the persistent volume so 12GB weights
    # are only downloaded once (first run) and reused on every subsequent run.
    os.environ["HF_HOME"]             = str(CACHE_PATH)
    os.environ["TRANSFORMERS_CACHE"]  = str(CACHE_PATH / "transformers")
    os.environ["HF_DATASETS_CACHE"]   = str(CACHE_PATH / "datasets")

    # Auth
    from huggingface_hub import login
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token, add_to_git_credential=False)
    print("✅ HuggingFace authenticated")

    # Apply DoRA patch before importing peft internals
    _patch_dora()

    # Unsloth MUST be imported before transformers
    from unsloth import FastVisionModel, is_bfloat16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    dataset_json = DATASET_PATH / "dataset_vision.json"
    dataset = _load_dataset(dataset_json)

    print("Loading Gemma-3-12B-IT vision model...")
    model, processor = FastVisionModel.from_pretrained(
        model_name="google/gemma-3-12b-it",
        load_in_4bit=True,   # QLoRA: 12B fits in 24GB with room for DoRA gradients
    )

    # Full Trinity on A10G:
    # - finetune_vision_layers=True  → A10G has 24GB; safe to adapt SigLIP too
    # - use_dora=True               → DoRA dtype bug patched above
    # - use_rslora=True             → stabilizes rank 64 training
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,      # A10G can handle vision adapter too
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=64,
        lora_alpha=32,
        lora_dropout=0,          # 0 = Unsloth fast-patches ALL layers
        bias="none",
        random_state=42,
        use_rslora=True,         # Stabilizes rank 64 — prevents gradient collapse
        use_dora=True,           # Weight decomposition: magnitude + direction
    )
    print("✅ Model loaded with full Trinity (QLoRA + DoRA + rsLoRA)")

    output_dir = str(OUTPUT_PATH / "adapters" / "trinity_a10g")

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,   # Effective batch = 8
            num_train_epochs=3,              # 3 passes over 287 examples
            learning_rate=2e-4,
            lr_scheduler_type="cosine",      # Cosine decay (better than linear for 3 epochs)
            warmup_ratio=0.1,
            optim="adamw_8bit",
            save_strategy="epoch",
            logging_steps=10,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),    # A10G supports bf16 → avoids fp16 hacks
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            seed=42,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=4096,             # A10G can handle full 4096 context
        ),
    )

    print("🚀 Starting training — 3 epochs × 287 examples...")
    trainer.train()

    print(f"✅ Training complete! Saving adapter to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Commit writes to the persistent volume — required for output_vol
    output_vol.commit()
    print(f"✅ Adapter committed to Modal Volume at {output_dir}")
    return output_dir


# ── Upload Dataset ────────────────────────────────────────────────────────────
@app.local_entrypoint()
def upload_dataset():
    """
    Upload dataset_vision.json + ui_screenshots/ to the Modal dataset volume.
    Run once (or when dataset changes):  modal run src/modal_train.py::upload_dataset
    """
    import os

    local_json       = Path("data/dataset_vision.json")
    local_screenshots = Path("data/ui_screenshots")

    assert local_json.exists(), f"Missing {local_json}"
    assert local_screenshots.exists(), f"Missing {local_screenshots}"

    screenshots = list(local_screenshots.glob("*.png"))
    print(f"Uploading dataset_vision.json + {len(screenshots)} screenshots (~107MB)...")
    with dataset_vol.batch_upload() as batch:
        batch.put_file(local_json, "dataset_vision.json")
        batch.put_directory(local_screenshots, "ui_screenshots")

    print(f"✅ Uploaded {len(screenshots)} images + dataset JSON to ghost-architect-dataset volume")


# ── Download Adapter ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def download_adapter():
    """
    Download the trained adapter from Modal output volume to local output/.
    Run after training:  modal run src/modal_train.py::download_adapter
    """
    local_out = Path("output/adapters/trinity_a10g")
    local_out.mkdir(parents=True, exist_ok=True)

    print("Listing adapter files in Modal output volume...")
    adapter_prefix = "adapters/trinity_a10g"
    files = list(output_vol.listdir(adapter_prefix, recursive=True))

    if not files:
        print("❌ No adapter files found. Has training completed?")
        return

    print(f"Downloading {len(files)} files...")
    for entry in files:
        dest = local_out / Path(entry.path).relative_to(adapter_prefix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with output_vol.read_file(entry.path) as f:
            dest.write_bytes(f.read())

    print(f"✅ Adapter downloaded to {local_out}/")
    print("Next step: python src/export.py --adapter_dir output/adapters/trinity_a10g --output_dir output/gguf")


# ── Run Training ─────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Default entrypoint: modal run src/modal_train.py"""
    result = train.remote()
    print(f"\n🎉 Training complete!")
    print(f"   Adapter is in Modal Volume at: {result}")
    print(f"   Download it: modal run src/modal_train.py::download_adapter")
