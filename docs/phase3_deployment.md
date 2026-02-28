# Phase 3: Deployment — GGUF Export, Ollama & Streamlit

## Overview

After training (Phase 1 text or Phase 2 vision), deploy the model locally via GGUF/Ollama and test with the Streamlit demo app or CLI inference.

**What you're building:**
- 📦 GGUF model file for Ollama (`src/export.py`)
- 🖥️ Ollama local inference
- 🌐 Streamlit web app for interactive demo (`src/app.py`)
- ⌨️ CLI inference with rich output (`src/inference.py`)

> **No FastAPI / no web API.** The project scope is: train → export GGUF → run locally with Ollama + Streamlit.

**Prerequisites:**
- ✅ Training complete: adapter weights in `output/adapters/`
- ✅ Ollama installed locally (`curl -fsSL https://ollama.com/install.sh | sh`)

---

## Step 1: Export to GGUF

```bash
python src/export.py
# Outputs to output/gguf/
```

Or use the Makefile:
```bash
make export
```

---

## Step 2: Register Model with Ollama

```bash
# Create the Modelfile
cat > Modelfile << 'EOF'
FROM ./output/gguf/ghost-architect-v1.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """You are Ghost Architect — an expert in UI analysis and PostgreSQL schema design.
When given a screenshot of a UI (dashboard, e-commerce, admin panel, etc.),
you output a complete, normalized PostgreSQL schema with proper data types,
primary keys, foreign keys, and indexes.
Output only valid SQL. No explanations unless asked."""
EOF

# Register and test
ollama create ghost-architect -f Modelfile
ollama run ghost-architect "Hello — are you ready?"
```

---

## Step 3: Interactive Demo (Streamlit)

```bash
streamlit run src/app.py
```

This launches a web UI where you can upload a screenshot and see the generated database schema.

---

## Step 4: CLI Testing

```bash
python src/inference.py
```

Rich terminal output for quick testing against the model.

---

## Step 5: Docker (Future)

The `docker/` directory is reserved for future containerized deployment. Currently empty.

---

## Deployment Checklist

- [ ] Adapter weights exist: `output/adapters/`
- [ ] GGUF exported: `python src/export.py`
- [ ] Ollama model registered: `ollama create ghost-architect -f Modelfile`
- [ ] Ollama tested: `ollama run ghost-architect`
- [ ] Streamlit demo works: `streamlit run src/app.py`
- [ ] CLI inference works: `python src/inference.py`

---

## Key Files

| File | Purpose |
|------|---------|
| `src/export.py` | GGUF export for Ollama |
| `src/app.py` | Streamlit web app (upload screenshot → schema) |
| `src/inference.py` | CLI testing with rich terminal output |

---

## Phase Dependency Map

```
Phase 1  (Text Trinity training)
    │
    ▼
Phase 2  (Vision training on 287 UI screenshots)
    │
    ▼
Phase 3  (Export GGUF + local deployment) ← YOU ARE HERE
```

---

## References

| Resource | Link |
|----------|------|
| Phase 1 training guide | `docs/phase1_trinity_training.md` |
| Phase 2 vision training guide | `docs/phase2_vision_training.md` |
| Full architecture | `docs/architecture.md` |
| Product requirements | `docs/prd.md` |
| Ollama Python client | https://github.com/ollama/ollama-python |
| GGUF format | https://github.com/ggerganov/llama.cpp |
