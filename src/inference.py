#!/usr/bin/env python3
"""
Ghost Architect — Inference
Test your trained model on a UI screenshot.

Usage (after training):
  python src/inference.py --adapter output/adapters/trinity_a10g
  python src/inference.py --adapter output/adapters/trinity_a10g --image data/ui_screenshots/paystack.co_44228.png
"""

import argparse
import random
import glob
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── SQL → Rich renderer ──────────────────────────────────────────────────────

def _parse_create_tables(sql: str) -> list[dict]:
    """
    Extract CREATE TABLE blocks from SQL text.
    Returns list of {"name": str, "columns": [{"name", "type", "constraints"}]}
    """
    tables = []
    # Match CREATE TABLE ... ( ... );  across multiple lines
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*?)\)\s*;",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(sql):
        table_name = match.group(1)
        body = match.group(2)

        columns = []
        for raw_line in body.split(","):
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            # Skip table-level constraints (PRIMARY KEY (...), FOREIGN KEY ..., UNIQUE ...)
            if re.match(r"(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)\s", line, re.IGNORECASE):
                continue
            # Parse: col_name  DATA_TYPE  [constraints...]
            parts = line.split()
            if not parts:
                continue
            col_name = parts[0].strip('`"')
            col_type = parts[1] if len(parts) > 1 else "?"
            # Collect remainder as constraints (PRIMARY KEY, NOT NULL, DEFAULT, REFERENCES)
            constraints = " ".join(parts[2:]) if len(parts) > 2 else ""
            columns.append({"name": col_name, "type": col_type, "constraints": constraints})

        if columns:
            tables.append({"name": table_name, "columns": columns})
    return tables


def _render_rich(image_name: str, sql: str, raw_sql: str):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich import box

    console = Console()

    console.print()
    console.print(Panel(
        f"[bold cyan]Ghost Architect[/bold cyan] — Schema Analysis\n"
        f"[dim]Image:[/dim] [yellow]{image_name}[/yellow]",
        box=box.DOUBLE_EDGE,
        style="bold",
    ))

    tables = _parse_create_tables(sql)

    if not tables:
        # Could not parse SQL — just pretty-print the raw output
        console.print(Panel(
            f"[white]{sql}[/white]",
            title="[bold green]Model Output[/bold green]",
            box=box.ROUNDED,
        ))
        return

    rendered_tables = []
    for t in tables:
        rich_table = Table(
            title=f"[bold magenta]{t['name']}[/bold magenta]",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold dim",
            min_width=40,
        )
        rich_table.add_column("Column",      style="bold white",  no_wrap=True)
        rich_table.add_column("Type",        style="cyan",        no_wrap=True)
        rich_table.add_column("Constraints", style="dim yellow",  no_wrap=False)

        for col in t["columns"]:
            name = col["name"]
            ctype = col["type"]
            cstr  = col["constraints"]

            # Emoji decoration
            if "PRIMARY KEY" in cstr.upper():
                icon = "🔑"
            elif "REFERENCES" in cstr.upper() or "FOREIGN" in cstr.upper():
                icon = "🔗"
            elif "NOT NULL" in cstr.upper():
                icon = " ●"
            else:
                icon = "  "

            rich_table.add_row(f"{icon} {name}", ctype, cstr)

        rendered_tables.append(Panel(rich_table, box=box.ROUNDED, style="dim"))

    # Print tables in columns (2 per row) for a professional layout
    console.print(Columns(rendered_tables, equal=True, expand=True))

    console.print(
        Panel(
            "[dim]🔑 Primary Key   🔗 Foreign Key   ● Not Null[/dim]",
            box=box.MINIMAL,
        )
    )


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(adapter_dir: str, image_path: str | None = None):
    import torch
    from unsloth import FastVisionModel
    from PIL import Image

    if image_path:
        img_file = Path(image_path)
    else:
        choices = glob.glob("data/ui_screenshots/*.png")
        if not choices:
            raise FileNotFoundError("No images found in data/ui_screenshots/")
        img_file = Path(random.choice(choices))

    logger.info(f"Loading Ghost Architect from {adapter_dir}...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=adapter_dir,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)

    logger.info(f"Analyzing: {img_file.name}")
    image = Image.open(img_file).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_file), "text": ""},
                {"type": "text",  "image": "",            "text": "Analyze the UI structure and generate an appropriate database schema."},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt, images=image, return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Generating schema...")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            use_cache=True,
            temperature=0.2,
            do_sample=True,
        )

    full_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    sql = full_text.split("model\n")[-1].strip() or full_text

    _render_rich(img_file.name, sql, sql)
    return sql


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default="output/adapters/trinity_a10g")
    parser.add_argument("--image",   type=str, default=None)
    args = parser.parse_args()
    run_inference(args.adapter, args.image)
