#!/usr/bin/env python3
"""
Production-Grade Gemma-3 Vision Fine-Tuning for UI-to-SQL
Includes Lazy-Loading to prevent RAM crashes on massive datasets.
"""

import argparse
import json
import torch
import os
from pathlib import Path
import logging

# Unsloth MUST be imported FIRST
from unsloth import FastVisionModel

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from dotenv import load_dotenv
from huggingface_hub import login

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- AUTHENTICATION ---
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    logger.info("✅ HuggingFace authenticated")
else:
    raise ValueError("HUGGINGFACE_TOKEN not in .env")

# --- DATASET LAZY LOADING ---
def load_vision_dataset(dataset_path: Path) -> Dataset:
    logger.info(f"Loading dataset from {dataset_path}...")
    
    raw = dataset_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw)
    
    valid_rows = []
    skipped = 0
    
    for item in parsed:
        img_path = Path(item.get("image_path") or item.get("image", ""))
        
        if not img_path.exists():
            skipped += 1
            continue
        
        # Embed image PATH as a string inside messages content.
        # UnslothVisionDataCollator._extract_images_videos_for_example() falls back to
        # process_vision_info(messages) when no top-level 'images' key is present.
        # fetch_image() then loads the PIL image via Image.open(path_string).
        #
        # All content items share a consistent Arrow schema {type, image, text}
        # with empty strings for unused fields — Arrow requires uniform schemas in lists.
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
        
        # Only 'messages' column — no 'image'/'images' top-level key.
        # This means TRL's _is_vision_dataset = False → VLM model check is skipped.
        # skip_prepare_dataset=True in SFTConfig ensures no text tokenization either.
        valid_rows.append({"messages": messages})
    
    logger.info(f"✅ Loaded {len(valid_rows)} valid examples (skipped {skipped})")
    return Dataset.from_list(valid_rows)

# --- TRINITY ARCHITECTURE ---
def create_vision_model():
    logger.info("Loading Gemma-3 vision model via Unsloth...")

    model, processor = FastVisionModel.from_pretrained(
        model_name="google/gemma-3-12b-it",
        load_in_4bit=True,
    )

    # FastVisionModel.get_peft_model (not peft.get_peft_model) — required so
    # Unsloth's custom CUDA kernels and memory optimizations apply correctly.
    model = FastVisionModel.get_peft_model(
        model,
        # SigLIP vision encoder is already pre-trained for visual feature extraction.
        # We only fine-tune the language model to map those features → SQL schemas.
        # Disabling vision layer LoRA also avoids a DoRA fp16/fp32 dtype mismatch:
        # DoRA passes x_eye in fp16 directly to fp32 lora_A without casting, unlike
        # regular LoRA which casts input first. SigLIP always runs in fp16 on T4.
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=64,           # High rank, stabilized by rsLoRA
        lora_alpha=32,
        lora_dropout=0,     # 0 = Unsloth fast-patches ALL layers (max speed)
        bias="none",
        random_state=42,
        use_rslora=True,    # Rank-stabilized (prevents gradient collapse at r=64)
        use_dora=False,     # DoRA crashes on T4: Unsloth's Gemma3 patch forces fp16 on
                            # q_proj input but DoRA passes it to fp32 lora_A without cast.
                            # Use Modal (modal_train.py) for full Trinity with DoRA.
    )

    logger.info(f"✅ Model loaded with Trinity adapters")
    return model, processor

# --- TRAINING LOOP ---
def train_vision_model(dataset_path: Path, output_dir: str):
    dataset = load_vision_dataset(dataset_path)
    model, processor = create_vision_model()
    
    # is_bfloat16_supported picks fp16 vs bf16 automatically based on GPU
    from unsloth import is_bfloat16_supported

    # SFTConfig (not TrainingArguments) required for Unsloth vision training.
    # dataset_text_field="" + skip_prepare_dataset tells SFTTrainer not to
    # treat this as a text dataset — the UnslothVisionDataCollator handles it.
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            optim="adamw_8bit",
            save_strategy="steps",
            save_steps=25,
            logging_steps=5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            seed=42,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )
    
    logger.info("Starting production training...")
    trainer.train()
    
    logger.info(f"✅ Training complete! Saving to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
    parser.add_argument("--output", type=str, default="output/adapters/vision_trinity")
    args = parser.parse_args()
    
    train_vision_model(args.dataset, args.output)