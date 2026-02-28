# AI Development Rules & Quality Control Guidelines

## Overview
This document establishes strict quality control rules and development guidelines for the Ghost Architect project to ensure consistency, reliability, and maintainability throughout the development process.

---

## 1. Code Quality Rules

### 1.1 Python Code Standards

**Mandatory Standards**:
- ✅ **PEP 8 Compliance**: All code must follow PEP 8 style guidelines
- ✅ **Type Hints**: All functions must include proper type annotations
- ✅ **Docstrings**: All classes and functions must have comprehensive docstrings
- ✅ **Error Handling**: Explicit exception handling for all potential failures
- ✅ **Logging**: Structured logging for debugging and monitoring

**Code Structure Rules**:
```python
# ✅ CORRECT: Proper function structure
def generate_schema(image_path: str, config: Dict[str, Any]) -> SchemaResult:
    """
    Generate database schema from UI screenshot.
    
    Args:
        image_path: Path to the UI screenshot
        config: Configuration parameters for generation
        
    Returns:
        SchemaResult containing generated SQL and metadata
        
    Raises:
        ValueError: If image_path is invalid
        ProcessingError: If schema generation fails
    """
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Implementation with proper error handling
        pass
    except Exception as e:
        logger.error(f"Schema generation failed: {str(e)}")
        raise ProcessingError(f"Failed to generate schema: {str(e)}")

# ❌ INCORRECT: Missing type hints and docstring
def generate_schema(image_path, config):
    # Implementation without proper structure
    pass
```

### 1.2 Testing Requirements

**Mandatory Testing**:
- ✅ **Unit Tests**: Minimum 80% code coverage for all modules
- ✅ **Integration Tests**: End-to-end testing of training and inference pipelines
- ✅ **Performance Tests**: Memory usage and speed benchmarks
- ✅ **Quality Tests**: SQL validity and relationship accuracy tests

**Test Structure Example**:
```python
# ✅ CORRECT: Comprehensive test structure
class TestSchemaGeneration:
    def test_valid_ui_screenshot(self):
        """Test schema generation with valid UI screenshot."""
        result = generate_schema("tests/data/valid_ui.png")
        assert result.sql_validity_score > 0.9
        assert len(result.tables) > 0
        assert all(table.has_primary_key for table in result.tables)
    
    def test_invalid_image_format(self):
        """Test handling of invalid image formats."""
        with pytest.raises(ValueError, match="Invalid image format"):
            generate_schema("tests/data/invalid.txt")
    
    def test_memory_usage(self):
        """Test memory usage stays within limits."""
        initial_memory = get_gpu_memory()
        generate_schema("tests/data/complex_ui.png")
        peak_memory = get_gpu_memory()
        assert (peak_memory - initial_memory) < 16000  # MB
```

---

## 2. Model Training Rules

### 2.1 Memory Management Rules

**Strict Memory Limits**:
- ✅ **T4 GPU Limit**: Never exceed 15.8GB VRAM usage (safety margin)
- ✅ **Memory Monitoring**: Continuous monitoring during training
- ✅ **OOM Recovery**: Automatic fallback procedures for memory issues
- ✅ **Gradient Accumulation**: Use accumulation instead of large batches

**Memory Validation**:
```python
# ✅ CORRECT: Memory monitoring implementation
class MemoryMonitor:
    def __init__(self, max_memory_gb: float = 15.8):
        self.max_memory_gb = max_memory_gb
        self.alert_threshold = max_memory_gb * 0.9  # Alert at 90%
    
    def check_memory_usage(self) -> bool:
        current_memory = torch.cuda.memory_allocated() / 1024**3
        if current_memory > self.alert_threshold:
            logger.warning(f"High memory usage: {current_memory:.2f}GB")
        if current_memory > self.max_memory_gb:
            raise MemoryError(f"Memory limit exceeded: {current_memory:.2f}GB")
        return current_memory < self.alert_threshold

# ❌ INCORRECT: No memory monitoring
def train_model():
    # Training without memory checks - PROHIBITED
    pass
```

### 2.2 Training Configuration Rules

**Configuration Validation**:
- ✅ **Parameter Validation**: All training parameters must be validated before training
- ✅ **Reproducibility**: Random seeds must be set for reproducible training
- ✅ **Checkpointing**: Regular checkpoints with validation
- ✅ **Early Stopping**: Implement early stopping to prevent overfitting

**Training Configuration Example**:
```python
# ✅ CORRECT: Validated training configuration
@dataclass
class TrainingConfig:
    model_name: str = "unsloth/gemma-3-12b-it-bnb-4bit"
    max_seq_length: int = 4096
    lora_rank: int = 64
    lora_alpha: int = 32
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 60
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.lora_rank > 64:
            raise ValueError("LoRA rank too high for T4 GPU")
        if self.batch_size > 1:
            raise ValueError("Batch size must be 1 for 12B model on T4")
        if self.max_seq_length > 4096:
            raise ValueError("Sequence length too long for memory constraints")
```

---

## 3. Data Quality Rules

### 3.1 Dataset Validation Rules

**Mandatory Validations**:
- ✅ **Format Validation**: All training data must match expected JSON schema
- ✅ **Content Quality**: Manual review of generated synthetic data
- ✅ **SQL Syntax**: All SQL schemas must be syntactically valid
- ✅ **Relationship Integrity**: Database relationships must be logically consistent

**Data Validation Implementation**:
```python
# ✅ CORRECT: Comprehensive data validation
class DatasetValidator:
    def validate_training_pair(self, image_path: str, sql_schema: str) -> bool:
        """Validate a single training pair."""
        checks = {
            "image_exists": os.path.exists(image_path),
            "image_valid": self.validate_image_format(image_path),
            "sql_syntax": self.validate_sql_syntax(sql_schema),
            "sql_completeness": self.check_schema_completeness(sql_schema),
            "relationship_integrity": self.validate_relationships(sql_schema)
        }
        
        failed_checks = [k for k, v in checks.items() if not v]
        if failed_checks:
            logger.error(f"Validation failed: {failed_checks}")
            return False
        
        return True
    
    def validate_sql_syntax(self, sql: str) -> bool:
        """Validate SQL syntax using sqlparse."""
        try:
            parsed = sqlparse.parse(sql)
            return len(parsed) > 0 and not any(token.ttype is sqlparse.tokens.Error 
                                             for token in parsed[0].flatten())
        except Exception:
            return False
```

### 3.2 Synthetic Data Generation Rules

**Quality Control for Synthetic Data**:
- ✅ **Teacher Model Validation**: Verify teacher model responses before storing
- ✅ **Diversity Requirements**: Ensure dataset covers variety of UI patterns
- ✅ **Quality Scoring**: Implement automated quality scoring for generated pairs
- ✅ **Human Review**: Random sampling for human validation

---

## 4. Application Interface Rules

### 4.1 Interface Design Rules

**Mandatory Standards**:
- ✅ **Input Validation**: Validate all inputs before processing
- ✅ **Error Handling**: Consistent error response format
- ✅ **Documentation**: Clear usage documentation

**Streamlit App Example** (`src/app.py`):
```python
# ✅ CORRECT: Proper input handling
import streamlit as st
from PIL import Image

uploaded = st.file_uploader("Upload UI screenshot", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    if uploaded.size > 10 * 1024 * 1024:  # 10MB limit
        st.error("Image too large (max 10MB)")
    else:
        image = Image.open(uploaded)
        # Process image and generate schema
        st.code(result, language="sql")
```

### 4.2 Security Rules

**Mandatory Security Measures**:
- ✅ **Input Sanitization**: Sanitize all user inputs
- ✅ **No Hardcoded Secrets**: Use environment variables for API keys
- ✅ **No Data Persistence**: Delete user images immediately after processing

---

## 5. Performance Rules

### 5.1 Response Time Requirements

**Performance Standards**:
- ✅ **Inference Latency**: <5 seconds per request via Ollama
- ✅ **Memory Efficiency**: <8GB GGUF model size
- ✅ **Resource Cleanup**: Automatic cleanup of temporary resources

**Performance Monitoring**:
```python
# ✅ CORRECT: Performance monitoring
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            
            # Alert if too slow
            if execution_time > 5.0:
                logger.warning(f"Slow execution: {func.__name__} took {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    return wrapper
```

---

## 6. Documentation Rules

### 6.1 Code Documentation Requirements

**Mandatory Documentation**:
- ✅ **API Documentation**: Complete OpenAPI specifications
- ✅ **Code Comments**: Explain complex logic and decisions
- ✅ **README Files**: Clear setup and usage instructions
- ✅ **Architecture Documentation**: System design documentation

### 6.2 Version Control Rules

**Git Workflow Rules**:
- ✅ **Commit Messages**: Clear, descriptive commit messages
- ✅ **Branch Naming**: Consistent branch naming convention
- ✅ **Code Reviews**: All changes require code review
- ✅ **Testing**: All commits must pass automated tests

**Commit Message Format**:
```bash
# ✅ CORRECT: Descriptive commit messages
feat: Add multimodal fusion layer for UI-to-SQL conversion
fix: Resolve memory leak in training checkpoint saving
docs: Update API documentation with error handling examples
test: Add performance tests for schema generation pipeline

# ❌ INCORRECT: Vague commit messages
update code
fix bug
more changes
```

---

## 7. Error Handling & Monitoring Rules

### 7.1 Error Handling Standards

**Mandatory Error Handling**:
- ✅ **Specific Exceptions**: Use specific exception types for different errors
- ✅ **Error Logging**: Log all errors with context
- ✅ **User-Friendly Messages**: Return helpful error messages to users
- ✅ **Recovery Procedures**: Implement graceful fallbacks where possible

```python
# ✅ CORRECT: Comprehensive error handling
class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass

class ModelLoadError(ProcessingError):
    """Exception for model loading failures."""
    pass

class SchemaGenerationError(ProcessingError):
    """Exception for schema generation failures."""
    pass

def generate_schema_safe(image_path: str) -> SchemaResult:
    """Generate schema with comprehensive error handling."""
    try:
        # Attempt schema generation
        result = generate_schema(image_path)
        return result
        
    except ModelLoadError as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(503, "Model temporarily unavailable")
        
    except SchemaGenerationError as e:
        logger.error(f"Schema generation failed: {str(e)}")
        raise HTTPException(422, "Unable to process this image")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Internal server error")
```

### 7.2 Monitoring & Alerting Rules

**Monitoring Requirements**:
- ✅ **System Metrics**: Monitor CPU, memory, and GPU usage
- ✅ **Application Metrics**: Track request rates, error rates, response times
- ✅ **Model Metrics**: Monitor model performance and accuracy
- ✅ **Alert Thresholds**: Set up alerts for critical issues

---

## 8. Deployment Rules

### 8.1 Deployment Standards

**Deployment Model**:
- ✅ **GGUF Export**: All models exported via `src/export.py` for Ollama
- ✅ **Local Inference**: Run via Ollama locally
- ✅ **Streamlit Demo**: Interactive demo via `src/app.py`
- ✅ **CLI Testing**: `src/inference.py` for quick validation

### 8.2 Release Process Rules

**Release Standards**:
- ✅ **Version Tagging**: Use semantic versioning for releases
- ✅ **Release Notes**: Document all changes in release notes
- ✅ **Rollback Procedures**: Have tested rollback procedures ready
- ✅ **Staged Deployment**: Deploy to staging before production

---

## 9. Quality Assurance Checklist

### 9.1 Pre-Commit Checklist
- [ ] All tests pass
- [ ] Code coverage >80%
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] Error handling implemented
- [ ] Performance within limits
- [ ] Security review completed

### 9.2 Pre-Release Checklist
- [ ] Full integration tests pass
- [ ] Performance benchmarks met
- [ ] Security audit completed
- [ ] Documentation updated
- [ ] GGUF export tested
- [ ] Ollama deployment verified

---

## 10. Violation Response Protocol

### 10.1 Rule Violation Handling

**When Rules Are Violated**:
1. **Immediate Fix**: Stop development and fix the violation
2. **Root Cause Analysis**: Understand why the violation occurred
3. **Process Improvement**: Update processes to prevent recurrence
4. **Documentation Update**: Update rules if necessary

### 10.2 Quality Gates

**No code proceeds without**:
- ✅ Passing all automated tests
- ✅ Meeting performance requirements
- ✅ Proper error handling implementation
- ✅ Complete documentation
- ✅ Security validation

---

These AI development rules ensure consistent, high-quality development throughout the Ghost Architect project lifecycle. All team members and contributors must follow these guidelines to maintain project integrity and reliability.