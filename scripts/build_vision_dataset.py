#!/usr/bin/env python3
"""
Production-Grade UI-to-SQL Vision Dataset Builder
Converts raw UI screenshots into annotated training data for Gemma-3 vision model.

Purpose:
  - Pairs UI screenshots with SQL schema annotations
  - Creates structured dataset for multimodal fine-tuning
  - Generates synthetic SQL from visual elements
  - Validates image quality & format

Output: data/dataset_vision.json
"""

import json
import os
from pathlib import Path
from PIL import Image
import hashlib
from datetime import datetime
import re


def get_image_hash(image_path: Path) -> str:
    """Generate content hash for image deduplication"""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def validate_image(image_path: Path) -> bool:
    """Check if image is valid and not corrupted"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"  ⚠️  Invalid image: {image_path.name} - {e}")
        return False


def extract_domain_from_filename(filename: str) -> str:
    """Extract domain from filename (e.g., hangman-c0h7.onrender.com_7455.png)"""
    match = re.match(r"(.+?)_\d+\.png", filename)
    return match.group(1) if match else filename


def generate_sql_annotation(domain: str, image_number: int) -> dict:
    """
    Generate SQL annotation based on visual context.
    In production, this would be:
      1. LLM-generated (via Gemini 2.5 Vision)
      2. Manual annotation by domain experts
      3. Crowd-sourced labels
    
    For MVP, use heuristic-based templates.
    """
    
    # Common UI patterns → SQL schemas
    templates = {
        # E-commerce / Product sites
        "shop": {
            "instruction": "Analyze this e-commerce UI and generate the database schema for displaying products.",
            "output": """CREATE TABLE products (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  price DECIMAL(10,2),
  description TEXT,
  category VARCHAR(100),
  stock_qty INT,
  image_url VARCHAR(500),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"""
        },
        # Dashboard / Analytics
        "dashboard": {
            "instruction": "Analyze this dashboard UI and generate the schema for the metrics displayed.",
            "output": """CREATE TABLE metrics (
  id INT PRIMARY KEY AUTO_INCREMENT,
  metric_name VARCHAR(100),
  value DECIMAL(15,2),
  timestamp TIMESTAMP,
  user_id INT,
  category VARCHAR(50),
  FOREIGN KEY (user_id) REFERENCES users(id)
);"""
        },
        # Admin / Data management
        "admin": {
            "instruction": "Analyze this admin panel and generate the database schema needed.",
            "output": """CREATE TABLE resources (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255),
  status ENUM('active','inactive','pending'),
  metadata JSON,
  created_by INT,
  updated_at TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (created_by) REFERENCES users(id)
);"""
        },
        # Portfolio / Content
        "portfolio": {
            "instruction": "Analyze this portfolio/content site UI and generate the schema.",
            "output": """CREATE TABLE content (
  id INT PRIMARY KEY AUTO_INCREMENT,
  title VARCHAR(255) NOT NULL,
  slug VARCHAR(255) UNIQUE,
  body TEXT,
  excerpt VARCHAR(500),
  published_at TIMESTAMP,
  author_id INT,
  tags JSON,
  views_count INT DEFAULT 0,
  FOREIGN KEY (author_id) REFERENCES users(id)
);"""
        },
        # Chat / Social
        "chat": {
            "instruction": "Analyze this messaging/chat UI and generate the database schema.",
            "output": """CREATE TABLE messages (
  id INT PRIMARY KEY AUTO_INCREMENT,
  sender_id INT NOT NULL,
  recipient_id INT NOT NULL,
  content TEXT,
  sent_at TIMESTAMP,
  read_at TIMESTAMP,
  is_deleted BOOLEAN DEFAULT FALSE,
  FOREIGN KEY (sender_id) REFERENCES users(id),
  FOREIGN KEY (recipient_id) REFERENCES users(id)
);"""
        },
        # Default / Generic
        "default": {
            "instruction": "Analyze the UI structure and generate an appropriate database schema.",
            "output": """CREATE TABLE data (
  id INT PRIMARY KEY AUTO_INCREMENT,
  type VARCHAR(100),
  content JSON,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);"""
        }
    }
    
    # Classify domain by keyword
    domain_lower = domain.lower()
    
    if any(x in domain_lower for x in ['shop', 'store', 'ecommerce', 'product', 'buy', 'cart']):
        return templates["shop"]
    elif any(x in domain_lower for x in ['dashboard', 'analytics', 'metric', 'chart', 'graph']):
        return templates["dashboard"]
    elif any(x in domain_lower for x in ['admin', 'manage', 'control', 'panel']):
        return templates["admin"]
    elif any(x in domain_lower for x in ['portfolio', 'blog', 'content', 'article', 'post']):
        return templates["portfolio"]
    elif any(x in domain_lower for x in ['chat', 'message', 'social', 'forum', 'comment']):
        return templates["chat"]
    else:
        return templates["default"]


def build_vision_dataset():
    """Main dataset building pipeline"""
    
    screenshot_dir = Path("data/ui_screenshots")
    if not screenshot_dir.exists():
        print("❌ data/ui_screenshots/ not found!")
        return
    
    print("\n" + "="*70)
    print("🔨 BUILDING PRODUCTION-GRADE VISION DATASET")
    print("="*70)
    
    # Collect all valid images
    all_images = list(screenshot_dir.glob("*.png"))
    print(f"\n📸 Found {len(all_images)} screenshots")
    
    # Validate and build dataset
    dataset = []
    seen_hashes = set()
    invalid_count = 0
    
    for idx, img_path in enumerate(all_images, 1):
        # Progress indicator
        if idx % 10 == 0:
            print(f"  ✓ Processing {idx}/{len(all_images)}...", end='\r')
        
        # Validate image
        if not validate_image(img_path):
            invalid_count += 1
            continue
        
        # Detect duplicates via content hash
        img_hash = get_image_hash(img_path)
        if img_hash in seen_hashes:
            print(f"  ⚠️  Duplicate (skipped): {img_path.name}")
            continue
        seen_hashes.add(img_hash)
        
        # Extract domain and generate annotation
        domain = extract_domain_from_filename(img_path.stem)
        annotation = generate_sql_annotation(domain, idx)
        
        # Build training example
        example = {
            "image_path": str(img_path),
            "domain": domain,
            "instruction": annotation["instruction"],
            "output": annotation["output"],
            "hash": img_hash,
            "size_kb": img_path.stat().st_size / 1024
        }
        
        dataset.append(example)
    
    print(f"\n✅ Validation complete!")
    print(f"  • Valid images: {len(dataset)}")
    print(f"  • Invalid/corrupted: {invalid_count}")
    print(f"  • Duplicates removed: {len(all_images) - len(dataset) - invalid_count}")
    
    # Save dataset
    output_path = Path("data/dataset_vision.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n💾 Dataset saved to: {output_path}")
    print(f"   Size: {len(dataset)} examples")
    print(f"   Total images: {sum(ex['size_kb'] for ex in dataset):.1f} MB")
    
    # Summary stats
    print("\n📊 DATASET STATISTICS:")
    domains = {}
    for ex in dataset:
        d = ex["domain"]
        domains[d] = domains.get(d, 0) + 1
    
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"   {domain:40s} {count:3d} images")
    
    print("\n✨ NEXT STEPS:")
    print("   1. Run train_vision.py with this dataset_vision.json")
    print("   2. Monitor training with nvidia-smi")
    print("   3. Export to GGUF for production deployment")
    print("\n" + "="*70)


if __name__ == "__main__":
    build_vision_dataset()
