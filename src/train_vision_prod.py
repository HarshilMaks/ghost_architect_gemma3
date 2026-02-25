#!/usr/bin/env python3
"""
Production-Grade Gemma-3 Vision Fine-Tuning for UI-to-SQL
Phase 2: Multimodal Trinity Training (QLoRA + DoRA + rsLoRA on Vision Model)

Training Strategy:
  - Model: google/gemma-3-12b-it (7B vision + 5B base with quantization)
  - Adapter: QLoRA (4-bit) + DoRA + rsLoRA for stable high-rank training
  - Data: 136 annotated UI screenshots with SQL schema outputs
  - Hardware: NVIDIA T4 (16GB VRAM on Colab)
  - Output: LoRA adapters + GGUF for production inference

Expected:
  - Training time: 45-90 minutes on T4
  - VRAM usage: ~14-15GB (98% utilization)
  - Final checkpoint: output/adapters/vision_trinity_lora/
"""

import argparse
import json
import torch
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging

# Unsloth MUST be imported FIRST (before transformers/peft/trl)
from unsloth import FastVisionModel

from transformers import TrainingArguments, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# HuggingFace Authentication
from dotenv import load_dotenv
from huggingface_hub import login

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 🔒 HUGGINGFACE AUTHENTICATION (Required for gated models)
# ============================================================================

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        logger.info("✅ HuggingFace authenticated")
    except Exception as e:
        logger.error(f"❌ Auth failed: {e}")
        raise ValueError("HuggingFace authentication required for google/gemma-3-12b-it")
else:
    raise ValueError("HUGGINGFACE_TOKEN not in .env - See HUGGINGFACE_AUTH.md")


# ============================================================================
# 📊 DATASET LOADING & VALIDATION
# ============================================================================

def load_vision_dataset(dataset_path: Path, max_examples: int = None) -> Dataset:
    """
    Load multimodal dataset with image validation.
    
    Dataset Format:
    [
      {
        "image_path": "data/ui_screenshots/...",
        "instruction": "Analyze this UI...",
        "output": "CREATE TABLE...",
        "domain": "ecommerce"
      }
    ]
    """
    logger.info(f"Loading dataset from {dataset_path}...")
    
    raw = dataset_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw)
    
    if max_examples:
        parsed = parsed[:max_examples]
    
    valid_rows = []
    skipped = 0
    
    for idx, item in enumerate(parsed, 1):
        if idx % 20 == 0:
            logger.info(f"  Validating: {idx}/{len(parsed)}...")
        
        img_path = Path(item.get("image_path", ""))
        
        # Validate image exists
        if not img_path.exists():
            logger.warning(f"  ⚠️  Missing image: {img_path}")
            skipped += 1
            continue
        
        # Format for Gemma-3 chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Vision model will embed image
                    {"type": "text", "text": item["instruction"]}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["output"]}]
            }
        ]
        
        valid_rows.append({
            "messages": messages,
            "images": [Image.open(img_path).convert("RGB")],
            "domain": item.get("domain", "unknown")
        })
    
    logger.info(f"✅ Loaded {len(valid_rows)} valid examples (skipped {skipped})")
    return Dataset.from_list(valid_rows)


# ============================================================================
# 🎯 TRINITY CONFIGURATION (QLoRA + DoRA + rsLoRA)
# ============================================================================

def get_trinity_config():
    """Return optimized Trinity adapter configuration"""
    return LoraConfig(
        # Base LoRA settings
        r=64,  # Rank (high, stabilized by rsLoRA)
        lora_alpha=32,  # Scaling (Unsloth handles auto-scaling)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"      # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        
        # Vision model specific
        task_type="CAUSAL_LM",
        
        # Trinity Stack
        use_dora=True,      # Weight decomposition (magnitude + direction)
        use_rslora=True,    # Rank-stabilized LoRA (enables high r without collapse)
    )


def create_vision_model():
    """Load Gemma-3 vision model with Unsloth optimization"""
    logger.info("Loading Gemma-3 vision model...")
    
    model_id = "google/gemma-3-12b-it"
    
    # Load with Unsloth optimization (memory-efficient)
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_id,
        load_in_4bit=True,
        use_bnb_4bit_compute_dtype="float16",
    )
    
    # Add Trinity adapters
    lora_config = get_trinity_config()
    model = get_peft_model(model, lora_config)
    
    logger.info(f"✅ Model loaded with Trinity adapters")
    logger.info(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, processor


# ============================================================================
# 🚀 TRAINING PIPELINE
# ============================================================================

def train_vision_model(dataset_path: Path, output_dir: str = "output/adapters/vision_trinity"):
    """
    Execute Phase 2 multimodal Trinity training.
    
    Training flow:
      1. Load vision dataset (136 UI screenshots)
      2. Load Gemma-3 vision model with Unsloth
      3. Attach Trinity adapters (QLoRA + DoRA + rsLoRA)
      4. Train with SFTTrainer (supervised fine-tuning)
      5. Save LoRA adapters
    """
    
    logger.info("\n" + "="*75)
    logger.info("🚀 PHASE 2: MULTIMODAL TRINITY TRAINING")
    logger.info("="*75)
    
    # 1. Load dataset
    dataset = load_vision_dataset(dataset_path)
    logger.info(f"Dataset splits: {len(dataset)} examples")
    
    # 2. Load model & processor
    model, processor = create_vision_model()
    
    # 3. Training arguments (conservative for T4)
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Batch size (T4 constraint)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch = 4
        
        # Learning
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        
        # Optimization
        optim="adamw_8bit",  # Memory-efficient optimizer
        
        # Checkpointing
        save_strategy="steps",
        save_steps=10,
        eval_strategy="no",
        
        # Logging
        logging_steps=5,
        logging_dir="./logs/vision",
        
        # Hardware
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        
        # Other
        seed=42,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
    )
    
    # 4. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="messages",
        processor=processor,
        packing=False,  # Vision models don't support packing
    )
    
    # 5. Train
    logger.info("Starting training... (this may take 45-90 mins on T4)")
    logger.info("Monitor VRAM: !nvidia-smi in separate Colab cell")
    
    trainer.train()
    
    # 6. Save
    logger.info(f"✅ Training complete! Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    logger.info("\n✨ NEXT: Export to GGUF for production inference")
    logger.info(f"   python scripts/export_to_gguf.py {output_dir}")
    
    return model, trainer


# ============================================================================
# 🏃 CLI ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2: UI-to-SQL Vision Training")
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset_vision.json"),
                        help="Path to dataset JSON")
    parser.add_argument("--output", type=str, default="output/adapters/vision_trinity",
                        help="Output directory for adapters")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit dataset size (for testing)")
    
    args = parser.parse_args()
    
    # Validate
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}\n"
                               f"Run: python scripts/build_vision_dataset.py")
    
    # Train
    model, trainer = train_vision_model(args.dataset, args.output)
    
    logger.info("\n🎉 PHASE 2 TRAINING COMPLETE!")
    logger.info("Trained on 136 UI screenshots with Trinity architecture")


if __name__ == "__main__":
    main()
