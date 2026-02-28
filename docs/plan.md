# Gemma-3 Progressive Implementation Plan

## Overview
This document provides a detailed step-by-step implementation plan for building a Gemma-3-12B fine-tuning system that progresses from basic text fine-tuning to the advanced "Ghost Architect" UI-to-SQL system.

## 📚 Understanding the Trinity Architecture (Learning Foundation)

### Why the Trinity? (QLoRA + DoRA + rsLoRA)

**The Memory Challenge:**
```
Standard Fine-Tuning (Gemma-3-12B):
- Model Parameters: 12 billion weights  
- Memory Needed: 48GB+ VRAM
- Cost: $2-3/hour on cloud GPUs

Your Reality:
- Available: Google Colab T4 (16GB VRAM)
- Cost: FREE
- Problem: 48GB doesn't fit in 16GB!
```

**The Trinity Solution:**
```
Layer 1: QLoRA (4-bit)     → Compresses 48GB → 12GB  (Memory: Model Weights)
Layer 2: LoRA (Rank 64)    → Trains 5% of params     (Memory: Gradients)  
Layer 3: rsLoRA            → Stabilizes high rank     (Quality: Training Stability)
Layer 4: DoRA              → Precision refinement     (Quality: Final Performance)

Result: 12B model training on 16GB GPU with near-full-precision performance!
```

### Trinity Components Explained

**QLoRA (The Compressor)** 🗜️
- **What**: 4-bit quantization using NF4 (NormalFloat4)
- **Why**: Shrinks model from 48GB → 12GB (75% reduction)
- **How**: Stores weights in 4-bit instead of 32-bit with minimal accuracy loss

**LoRA (The Adapter)** 🔌  
- **What**: Low-Rank Adaptation - only trains small "adapter" layers
- **Why**: Instead of updating 12B parameters, train only 100M adapter parameters
- **Mathematical**: `Output = Original_Weight + LoRA_A × LoRA_B`

**rsLoRA (The Stabilizer)** ⚖️
- **What**: Rank-Stabilized LoRA with improved scaling: `alpha/√r` instead of `alpha/r`
- **Why**: Enables high-rank training (rank 64) without gradient collapse
- **Impact**: 4x more learning capacity vs standard LoRA

**DoRA (The Precision Booster)** 🎯
- **What**: Weight-Decomposed Adaptation - separates magnitude and direction
- **Why**: More precise weight updates, reduces quantization errors
- **Impact**: 2-5% performance improvement on complex tasks

---

## Phase 1: Foundation - Trinity Architecture (Text Fine-Tuning)
*Duration: 2-3 weeks*
*Goal: Production-ready text fine-tuning system with QLoRA + DoRA + rsLoRA*

### 1.1 Environment Setup & Validation
**Objective**: Establish reliable development environment and validate hardware compatibility

**Steps**:
1. **Google Colab Pro Setup**
   - Subscribe to Colab Pro for T4 GPU access
   - Use the project notebook: `notebooks/main.ipynb`
   - Verify T4 GPU allocation: `!nvidia-smi`

2. **Dependency Installation**
   ```bash
   # Install core dependencies
   pip install --upgrade pip setuptools wheel
   pip install "unsloth==2026.1.4"
   pip install "trl>=0.18.2,<=0.24.0,!=0.19.0" peft accelerate bitsandbytes
   pip install torch>=2.1.0 transformers>=4.38.0

   # Note: do not force-install xformers on Colab T4; if no matching wheel exists
   # pip will attempt a source build and fail.
   ```

3. **Environment Validation Script**
   - Run: `!python scripts/validate_environment.py`
   - Confirm dependency and GPU checks pass

4. **Memory Validation Test**
   - Load Gemma-3-12B in 4-bit quantization
   - Monitor VRAM usage: Target <16GB
   - Document baseline memory consumption

**Deliverables**:
- [ ] Working Colab environment with T4 access
- [ ] All dependencies installed without conflicts
- [ ] Memory baseline documented
- [ ] Environment validation checklist

---

### 1.2 Trinity Training Implementation
**Objective**: Implement the QLoRA + DoRA + rsLoRA training pipeline

**Steps**:
1. **Create Training Script Architecture**
   ```python
   # File: src/train.py
   # Structure:
   - Model loading with 4-bit quantization
   - Trinity configuration (QLoRA + DoRA + rsLoRA)
   - Training loop with memory monitoring
   - Checkpoint management
   - Loss tracking and validation
   ```

2. **Trinity Configuration Details**
   ```python
   # Model Loading Configuration
   model_name = "unsloth/gemma-3-12b-it-bnb-4bit"  # Pre-quantized for speed
   max_seq_length = 4096  # Max safe length for 16GB VRAM (DO NOT EXCEED!)
   load_in_4bit = True    # QLoRA activation
   
   # LoRA Configuration (The Heart of the Trinity)
   r = 64                    # Rank (learning capacity) - sweet spot for 16GB VRAM
   lora_alpha = 32          # Scaling factor (Unsloth auto-adjusts)
   target_modules = [        # Which layers to adapt (70% of model parameters)
       "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
       "gate_proj", "up_proj", "down_proj"      # MLP layers
   ]
   use_rslora = True        # CRITICAL for rank 64! Prevents training divergence
   use_dora = True          # +2-5% performance boost (costs +1.5GB VRAM)
   
   # Training Arguments
   per_device_train_batch_size = 1      # MUST be 1 for 12B on T4
   gradient_accumulation_steps = 4      # Simulates batch_size=4
   learning_rate = 2e-4                 # LoRA sweet spot (proven across papers)
   optimizer = "adamw_8bit"             # 75% VRAM savings vs standard AdamW
   max_steps = 60                       # Prevents overtraining on large datasets
   ```

3. **Memory Optimization Strategy**
   ```python
   # Target Memory Usage: 15.6GB on T4 GPU (16GB total)
   memory_allocation = {
       "model_weights_4bit": 7.6,      # GB - Quantized model
       "gradients_rank64": 5.5,        # GB - High-rank LoRA gradients  
       "context_overhead": 2.5,        # GB - 4096 token context
       "total": 15.6                   # GB - 98% GPU utilization
   }
   ```

4. **OOM Recovery Protocol**
   ```python
   # If training crashes with "CUDA Out of Memory", apply fixes in order:
   
   # Fix #1: Reduce Context Length (Frees 3GB)
   max_seq_length = 2048  # Was 4096
   
   # Fix #2: Lower Rank (Frees 2GB)  
   r = 32  # Was 64
   
   # Fix #3: Disable DoRA (Frees 1.5GB)
   use_dora = False  # Was True
   
   # Fix #4: Target Fewer Modules (Frees 2GB)
   target_modules = ["q_proj", "v_proj"]  # Was 7 modules
   
   # Apply fixes sequentially until training works!
   ```

**Deliverables**:
- [ ] Complete `src/train.py` with Trinity implementation
- [ ] Memory monitoring dashboard
- [ ] OOM recovery system tested
- [ ] Training configuration documented

---

### 1.3 Dataset Creation & Preparation
**Objective**: Create high-quality training dataset for text fine-tuning

#### **Choose Your Use Case (Phase 1 Foundation)**

**For Phase 1, pick ONE specialization to master the Trinity architecture:**

#### **Implementation Steps**:
**Objective**: Create high-quality training dataset for text fine-tuning

**Steps**:
1. **Dataset Format Design**
   ```json
   {
     "conversations": [
       {
         "instruction": "Clear, specific task instruction",
         "input": "Optional context or input data", 
         "output": "Expected model response"
       }
     ]
   }
   ```

2. **Data Collection Strategy**
   - **Option A**: Use existing instruction datasets (Alpaca, OpenAssistant)
   - **Option B**: Create domain-specific dataset for your use case
   - **Option C**: Hybrid approach with curated examples

3. **Data Quality Assurance**
   - Minimum 50 high-quality examples for initial testing
   - Target 1000+ examples for production training
   - Instruction diversity validation
   - Response quality scoring

4. **Data Preprocessing Pipeline**
   - Validate dataset format with `scripts/validate_dataset.py`
   - Tokenization length checking (max 4096)
   - Duplicate detection and removal

**Deliverables**:
- [ ] `data/dataset.json` with validated training data
- [ ] Data preprocessing pipeline
- [ ] Quality assurance metrics
- [ ] Data validation tests

---

### 1.4 Training Execution & Monitoring
**Objective**: Execute training with full monitoring and validation

**Steps**:
1. **Pre-training Validation**
   - Model loads without OOM
   - Dataset processes correctly
   - All configurations validated
   - Baseline inference test

2. **Training Execution**
   ```python
   # Key training parameters:
   max_steps = 60  # Adjust based on dataset size
   learning_rate = 2e-4
   warmup_steps = 10
   logging_steps = 1
   save_steps = 20
   ```

3. **Real-time Monitoring**
   - GPU memory usage tracking
   - Loss convergence monitoring
   - Training speed (tokens/second)
   - Temperature and perplexity metrics

4. **Validation During Training**
   - Generate sample responses every 20 steps
   - Monitor overfitting indicators
   - Validate checkpoint integrity

**Deliverables**:
- [ ] Trained LoRA adapter saved in `output/adapters/`
- [ ] Training logs and metrics
- [ ] Validation results documented
- [ ] Performance benchmarks recorded

---

### 1.5 Export & Deployment Pipeline
**Objective**: Convert trained model to GGUF format for production deployment

**Steps**:
1. **LoRA Adapter Validation**
   ```python
   # Test adapter loading and inference
   from peft import AutoPeftModelForCausalLM
   model = AutoPeftModelForCausalLM.from_pretrained("output/adapters/")
   # Validate responses quality
   ```

2. **GGUF Conversion**
   ```python
   # File: src/export.py
   model.save_pretrained_gguf("output/gguf/", tokenizer, quantization_method="q4_k_m")
   ```

3. **Ollama Integration**
   ```bash
   # Create Modelfile
   FROM ./output/gguf/model.gguf
   TEMPLATE """{{ .Prompt }}"""
   PARAMETER temperature 0.7
   ```

4. **Production Testing**
   - Load model in Ollama: `ollama create gemma3-trinity -f Modelfile`
   - Performance testing: latency, throughput
   - Quality evaluation on test set

**Deliverables**:
- [ ] GGUF model file ready for deployment
- [ ] Ollama Modelfile configured
- [ ] Production performance metrics
- [ ] Deployment documentation

---

## Phase 2: Vision Specialization - Ghost Architect (UI-to-SQL)
*Duration: 3-4 weeks*
*Goal: Multimodal system for UI screenshot to database schema conversion*

### 2.1 Vision Training Setup
**Objective**: Train Gemma-3 on UI screenshots for SQL schema generation

**Steps**:
1. **Two Training Paths**
   - **Modal A10G** (`src/modal_train.py`): Full Trinity (QLoRA+DoRA+rsLoRA), 3 epochs, 4096 ctx
   - **Colab T4** (`src/train_vision.py`): QLoRA+rsLoRA only, 1 epoch, 2048 ctx

2. **Vision Dataset**
   - 287 UI screenshot–SQL pairs in `data/dataset_vision.json`
   - Built by `scripts/build_vision_dataset.py` using Gemini API (`src/synthetic_generator.py`)
   - Image paths embedded in messages (no top-level `images` column)

**Deliverables**:
- [ ] Vision training completes on Modal or Colab
- [ ] Adapter weights saved to `output/adapters/`
- [ ] GGUF exported via `src/export.py`

---

### 2.2 Synthetic Dataset Generation
**Objective**: Create UI screenshot–SQL pairs for vision training

**Steps**:
1. **Screenshot Collection**
   - `scripts/download_datasets.py` — Playwright scraper for UI screenshots
   - 287 PNGs in `data/ui_screenshots/`

2. **SQL Annotation**
   - `src/synthetic_generator.py` — Uses Gemini API (`google-generativeai`) to generate SQL from screenshots
   - `scripts/build_vision_dataset.py` — Orchestrates dataset creation

3. **Output**
   - `data/dataset_vision.json` — 287 annotated examples

**Deliverables**:
- [ ] 287+ validated UI-SQL pairs created
- [ ] Dataset format validated for TRL/Unsloth compatibility

---

### 2.3 Deployment & Demo
**Objective**: Export model and provide interactive testing

**Steps**:
1. **GGUF Export**
   - `src/export.py` converts adapters to GGUF format
   - Register with Ollama for local inference

2. **Interactive Demo**
   - `src/app.py` — Streamlit web app (upload screenshot → see schema)
   - `src/inference.py` — CLI testing with rich terminal output

3. **Testing**
   - Validate generated SQL quality on held-out examples
   - Compare Modal vs Colab training quality

**Deliverables**:
- [ ] GGUF model exported to `output/gguf/`
- [ ] Ollama model registered and tested
- [ ] Streamlit demo functional
- [ ] CLI inference working

---

## Post-Implementation: Testing & Validation

### System Integration Testing
1. **End-to-End Validation**
   - UI screenshot → SQL schema generation
   - Schema quality evaluation
   - Performance benchmarking

2. **User Acceptance Testing**
   - Test with real-world UI examples
   - Validate against actual database schemas
   - Collect user feedback

3. **Production Readiness**
   - Load testing and performance optimization
   - Security audit and vulnerability assessment
   - Documentation and user guides

---

## Phase Completion Criteria

### Phase 1 Complete When:
- [ ] Trinity training pipeline works reliably on T4 GPU
- [ ] Model exports to GGUF and runs in Ollama
- [ ] Memory optimization protocols validated
- [ ] Complete documentation and reproducibility

### Phase 2 Complete When:
- [ ] Ghost Architect generates valid SQL from UI screenshots
- [ ] GGUF exported and runs in Ollama
- [ ] Streamlit demo and CLI inference functional
- [ ] Quality metrics meet expectations

---

## Success Metrics

### Technical Metrics
- **Memory Efficiency**: <16GB VRAM usage on T4
- **Training Speed**: >100 tokens/second during fine-tuning
- **Model Quality**: Comparable to baseline on standard benchmarks
- **SQL Accuracy**: >90% syntactically correct schemas from UI
- **Inference**: <5 second response time via Ollama

### Business Metrics
- **Innovation Value**: First open-source UI-to-SQL system
- **Market Readiness**: Production-deployable solution
- **Learning Outcomes**: Deep understanding of multimodal LLM architecture
- **Scalability**: System handles increasing complexity and load
