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

from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset, Image as DatasetsImage

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
        
        # Format for Gemma-3
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
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
            # CRITICAL FIX: Pass the string path, NOT the loaded PIL Image
            "images": [str(img_path)]
        })
    
    logger.info(f"✅ Loaded {len(valid_rows)} valid examples (skipped {skipped})")
    
    # Create the dataset
    ds = Dataset.from_list(valid_rows)
    
    # CRITICAL FIX: Cast the column to HuggingFace Image type for Lazy Loading
    # This prevents your RAM from exploding when training on hundreds of images
    ds = ds.cast_column("images", DatasetsImage())
    
    return ds

# --- TRINITY ARCHITECTURE ---
def get_trinity_config():
    return LoraConfig(
        r=64, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,      
        use_rslora=True,    
    )

def create_vision_model():
    logger.info("Loading Gemma-3 vision model via Unsloth...")
    
    model, processor = FastVisionModel.from_pretrained(
        model_name="google/gemma-3-12b-it",
        load_in_4bit=True,
        use_bnb_4bit_compute_dtype="float16",
    )
    
    model = get_peft_model(model, get_trinity_config())
    logger.info(f"✅ Model loaded with Trinity adapters")
    return model, processor

# --- TRAINING LOOP ---
def train_vision_model(dataset_path: Path, output_dir: str):
    dataset = load_vision_dataset(dataset_path)
    model, processor = create_vision_model()
    
    # Format the dataset correctly for Unsloth Vision
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    
    processor = get_chat_template(
        processor,
        chat_template="gemma",
    )
    
    def formatting_prompts_func(examples):
        texts = [processor.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]
        return {"text": texts, "images": examples["images"]}
    
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    training_args = TrainingArguments(
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
        remove_unused_columns=False, # Required for vision models
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=processor,
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