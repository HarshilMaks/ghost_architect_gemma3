"""Validate dataset.json is present, non-empty, and valid JSON."""
import json
import sys
from pathlib import Path

DATASET = Path("data/dataset.json")

if not DATASET.exists():
    print(f"ERROR: {DATASET} not found. Add your training data first.")
    sys.exit(1)

raw = DATASET.read_text().strip()
if not raw:
    print(f"ERROR: {DATASET} is empty. Add training examples before running.")
    sys.exit(1)

try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    print(f"ERROR: {DATASET} is not valid JSON: {e}")
    sys.exit(1)

if not isinstance(data, list):
    print(f"ERROR: {DATASET} must contain a JSON array of examples.")
    sys.exit(1)

print(f"OK: {DATASET} — {len(data)} training example(s), valid JSON.")
