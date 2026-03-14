#!/usr/bin/env python3
"""
Ghost Architect — Streamlit Web App
Upload a UI evidence pack (multi-screenshot) -> AI generates PostgreSQL schema.

Run locally (after modal run ::download_adapter):
  streamlit run src/app.py

Run on Colab (after training completes):
  See notebooks/main.ipynb Cell 18 (tunnel cell)
"""

import re
import io
from uuid import uuid4
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Ghost Architect",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)

MIN_REQUIRED_IMAGES = 3
MAX_RECOMMENDED_IMAGES = 6

# ── Sidebar — adapter path ────────────────────────────────────────────────────
with st.sidebar:
    st.title("👻 Ghost Architect")
    st.markdown("**UI → PostgreSQL Schema**")
    st.divider()

    adapter_dir = st.text_input(
        "Adapter path",
        value="output/adapters/trinity_a10g",
        help="Path to your trained LoRA adapter directory",
    )
    st.caption("Change this to `output/adapters/phase2` if you trained on Colab.")
    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload **3-6 screenshots** from the same product flow\n"
        "2. Include: list/table view + create/edit form + detail/dashboard\n"
        "3. Model analyses each image separately\n"
        "4. App merges evidence into one consolidated PostgreSQL schema"
    )
    st.info(
        "Precision mode: tables/columns are included only when supported by visible UI evidence."
    )


# ── SQL Parser (shared with inference.py) ────────────────────────────────────
def _split_sql_columns(body: str) -> list[str]:
    """Split CREATE TABLE body by top-level commas (keeps DECIMAL(10,2) intact)."""
    parts = []
    current = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_create_tables(sql: str) -> list[dict]:
    """Extract CREATE TABLE blocks → list of {name, columns}."""
    tables = []
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*?)\)\s*;",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(sql):
        table_name = match.group(1)
        body = match.group(2)
        columns = []
        for raw_line in _split_sql_columns(body):
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            if re.match(r"(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)\s", line, re.IGNORECASE):
                continue
            parts = line.split()
            if not parts:
                continue
            col_name    = parts[0].strip('`"')
            col_type    = parts[1] if len(parts) > 1 else "?"
            constraints = " ".join(parts[2:]) if len(parts) > 2 else ""
            columns.append({"name": col_name, "type": col_type, "constraints": constraints})
        if columns:
            tables.append({"name": table_name, "columns": columns})
    return tables


def _merge_constraints(existing: str, new: str) -> str:
    existing = existing.strip()
    new = new.strip()
    if not existing:
        return new
    if not new:
        return existing
    merged_tokens = []
    for token in f"{existing} {new}".split():
        if token not in merged_tokens:
            merged_tokens.append(token)
    return " ".join(merged_tokens)


def merge_table_sets(table_sets: list[list[dict]]) -> list[dict]:
    """Merge tables from multiple screenshots into a single consolidated schema."""
    merged: dict[str, dict] = {}
    for tables in table_sets:
        for table in tables:
            table_key = table["name"].lower()
            if table_key not in merged:
                merged[table_key] = {
                    "name": table["name"],
                    "columns_by_key": {},
                    "column_order": [],
                }
            table_entry = merged[table_key]
            for col in table["columns"]:
                col_key = col["name"].lower()
                existing = table_entry["columns_by_key"].get(col_key)
                if existing is None:
                    table_entry["columns_by_key"][col_key] = {
                        "name": col["name"],
                        "type": col["type"],
                        "constraints": col["constraints"],
                    }
                    table_entry["column_order"].append(col_key)
                    continue

                if existing["type"] in {"?", "TEXT"} and col["type"] not in {"?", "TEXT"}:
                    existing["type"] = col["type"]
                if len(col["constraints"].strip()) > len(existing["constraints"].strip()):
                    existing["constraints"] = col["constraints"]
                else:
                    existing["constraints"] = _merge_constraints(
                        existing["constraints"], col["constraints"]
                    )

    result = []
    for table_key in sorted(merged):
        table_entry = merged[table_key]
        columns = [
            table_entry["columns_by_key"][col_key]
            for col_key in table_entry["column_order"]
        ]
        result.append({"name": table_entry["name"], "columns": columns})
    return result


def _normalize_identifier(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    return normalized.strip("_") or "unnamed_entity"


def tables_to_sql(tables: list[dict]) -> str:
    """Render merged table representation back into SQL CREATE TABLE statements."""
    statements = []
    for table in tables:
        table_name = _normalize_identifier(table["name"])
        column_lines = []
        for col in table["columns"]:
            col_name = _normalize_identifier(col["name"])
            col_type = col["type"] if col["type"] and col["type"] != "?" else "TEXT"
            constraints = col["constraints"].strip()
            suffix = f" {constraints}" if constraints else ""
            column_lines.append(f"    {col_name} {col_type}{suffix}")
        statements.append(
            f"CREATE TABLE {table_name} (\n" + ",\n".join(column_lines) + "\n);"
        )
    return "\n\n".join(statements)


def render_schema_cards(tables: list[dict]):
    """Render parsed tables as styled Streamlit columns."""
    if not tables:
        return False

    # Layout: up to 3 tables per row
    chunk_size = 3
    for i in range(0, len(tables), chunk_size):
        row_tables = tables[i : i + chunk_size]
        cols = st.columns(len(row_tables))
        for col, table in zip(cols, row_tables):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        background:#1e1e2e;
                        border:1px solid #7c3aed;
                        border-radius:8px;
                        padding:12px;
                        margin-bottom:12px;
                    ">
                    <h4 style="color:#a78bfa;margin:0 0 8px 0;">📋 {table['name']}</h4>
                    """,
                    unsafe_allow_html=True,
                )
                for col_def in table["columns"]:
                    cname = col_def["name"]
                    ctype = col_def["type"]
                    cstr  = col_def["constraints"].upper()

                    if "PRIMARY KEY" in cstr:
                        icon, color = "🔑", "#fbbf24"
                    elif "REFERENCES" in cstr or "FOREIGN" in cstr:
                        icon, color = "🔗", "#60a5fa"
                    elif "NOT NULL" in cstr:
                        icon, color = "●", "#34d399"
                    else:
                        icon, color = "○", "#9ca3af"

                    st.markdown(
                        f"<p style='margin:2px 0;font-size:13px;'>"
                        f"<span style='color:{color}'>{icon}</span> "
                        f"<span style='color:#e2e8f0'>{cname}</span> "
                        f"<span style='color:#64748b'>{ctype}</span>"
                        f"</p>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
    return True


# ── Model loader (cached — loads once, reused across uploads) ─────────────────
@st.cache_resource(show_spinner=False)
def load_model(adapter_path: str):
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(
        model_name=adapter_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    return model, processor


def generate_schema(model, processor, image: Image.Image, image_path_str: str) -> str:
    import torch

    strict_instruction = (
        "Analyze the UI and output only PostgreSQL CREATE TABLE statements. "
        "Only include entities, columns, and relationships that are clearly visible in the screenshot. "
        "If uncertain, omit it instead of guessing. "
        "Prefer snake_case names and include PRIMARY KEY / FOREIGN KEY constraints when evidence exists."
    )

    # IMPORTANT: message format MUST match train_vision.py structure
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path_str, "text": ""},
                {"type": "text", "image": "", "text": strict_instruction},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,
            use_cache=True,
            do_sample=False,
        )

    full_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return full_text.split("model\n")[-1].strip() or full_text


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#a78bfa;'>👻 Ghost Architect</h1>"
    "<p style='text-align:center;color:#64748b;'>Fine-tuned Gemma-3-12B · Multi-image UI Evidence -> PostgreSQL Schema</p>",
    unsafe_allow_html=True,
)
st.divider()

col_upload, col_schema = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("① Upload UI Evidence Pack")
    uploaded_files = st.file_uploader(
        "Upload 3-6 PNG/JPG screenshots from the same product",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.caption(f"Uploaded: {len(uploaded_files)} screenshot(s)")

        if len(uploaded_files) < MIN_REQUIRED_IMAGES:
            st.warning(
                "Precision mode needs at least 3 screenshots: "
                "list/table page + create/edit form + detail/dashboard."
            )
        elif len(uploaded_files) > MAX_RECOMMENDED_IMAGES:
            st.info(
                "You can continue, but 3-6 screenshots usually gives the best quality/speed balance."
            )

        preview_cols = st.columns(min(3, len(uploaded_files)))
        for idx, uploaded in enumerate(uploaded_files):
            image_bytes = uploaded.getvalue()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            with preview_cols[idx % len(preview_cols)]:
                st.image(image, caption=uploaded.name, use_container_width=True)

with col_schema:
    st.subheader("② Generated Schema")

    if not uploaded_files:
        st.info("Upload your UI evidence pack on the left to get started.")
    elif len(uploaded_files) < MIN_REQUIRED_IMAGES:
        st.info(
            "Add at least 3 screenshots for precise schema generation."
        )
    else:
        generate_btn = st.button(
            "🚀 Generate Precise Architecture",
            type="primary",
            use_container_width=True,
        )

        if generate_btn:
            with st.spinner("Loading model..."):
                try:
                    model, processor = load_model(adapter_dir)
                except Exception as e:
                    st.error(f"Could not load adapter from `{adapter_dir}`\n\n{e}")
                    st.stop()

            table_sets = []
            per_image_sql = []
            progress = st.progress(0)
            status = st.empty()

            for index, uploaded in enumerate(uploaded_files, start=1):
                status.write(f"Analyzing {uploaded.name} ({index}/{len(uploaded_files)})...")
                image_bytes = uploaded.getvalue()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                safe_name = Path(uploaded.name).name.replace(" ", "_")
                tmp_path = Path("/tmp") / f"ghost_architect_{uuid4().hex}_{safe_name}"
                tmp_path.write_bytes(image_bytes)
                try:
                    sql_candidate = generate_schema(model, processor, image, str(tmp_path))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

                parsed_tables = parse_create_tables(sql_candidate)
                table_sets.append(parsed_tables)
                per_image_sql.append((uploaded.name, sql_candidate, len(parsed_tables)))
                progress.progress(index / len(uploaded_files))

            status.empty()
            merged_tables = merge_table_sets(table_sets)
            consolidated_sql = tables_to_sql(merged_tables)

            st.success(
                f"Consolidated schema generated from {len(uploaded_files)} screenshots."
            )

            # ── Visual schema cards ──────────────────────────────────────────
            if merged_tables:
                st.markdown(
                    "<p style='color:#64748b;font-size:12px;'>"
                    "🔑 Primary Key &nbsp; 🔗 Foreign Key &nbsp; ● Not Null</p>",
                    unsafe_allow_html=True,
                )
                render_schema_cards(merged_tables)
            else:
                st.warning("Could not build a consolidated schema — showing per-image outputs below.")

            # ── Raw SQL (always shown, collapsible) ──────────────────────────
            with st.expander("View consolidated SQL", expanded=(not merged_tables)):
                st.code(consolidated_sql or "-- No consolidated SQL produced", language="sql")

            with st.expander("View per-image model outputs"):
                for file_name, sql_candidate, table_count in per_image_sql:
                    st.markdown(f"**{file_name}** — parsed tables: {table_count}")
                    st.code(sql_candidate, language="sql")
