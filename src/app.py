#!/usr/bin/env python3
"""
Ghost Architect — Streamlit Web App
Upload a UI screenshot → AI generates PostgreSQL schema → visualized live.

Run locally (after modal run ::download_adapter):
  streamlit run src/app.py

Run on Colab (after training completes):
  See notebooks/main.ipynb Cell 18 (tunnel cell)
"""

import sys
import os
import re
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
        "1. Upload any UI screenshot\n"
        "2. Gemma-3-12B (fine-tuned) analyses the visual layout\n"
        "3. Generates PostgreSQL `CREATE TABLE` statements\n"
        "4. Schema is displayed as a professional ER diagram"
    )


# ── SQL Parser (shared with inference.py) ────────────────────────────────────
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
        for raw_line in body.split(","):
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

    # IMPORTANT: message format MUST match train_vision.py exactly
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path_str, "text": ""},
                {"type": "text",  "image": "",             "text": "Analyze the UI structure and generate an appropriate database schema."},
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
            temperature=0.2,
            do_sample=True,
        )

    full_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return full_text.split("model\n")[-1].strip() or full_text


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#a78bfa;'>👻 Ghost Architect</h1>"
    "<p style='text-align:center;color:#64748b;'>Fine-tuned Gemma-3-12B · UI Screenshot → PostgreSQL Schema</p>",
    unsafe_allow_html=True,
)
st.divider()

col_upload, col_schema = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("① Upload UI Screenshot")
    uploaded = st.file_uploader(
        "PNG or JPG — any SaaS dashboard, admin panel, or web app",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, use_container_width=True)

with col_schema:
    st.subheader("② Generated Schema")

    if not uploaded:
        st.info("Upload a screenshot on the left to get started.")
    else:
        generate_btn = st.button("🚀 Generate Architecture", type="primary", use_container_width=True)

        if generate_btn:
            # Save upload to a temp file so the model can reference it by path
            tmp_path = Path("/tmp") / uploaded.name
            tmp_path.write_bytes(uploaded.getvalue())

            with st.spinner("Loading model..."):
                try:
                    model, processor = load_model(adapter_dir)
                except Exception as e:
                    st.error(f"Could not load adapter from `{adapter_dir}`\n\n{e}")
                    st.stop()

            with st.spinner("Analysing visual layout and generating SQL..."):
                sql = generate_schema(model, processor, image, str(tmp_path))

            st.success("Schema generated!")

            # ── Visual schema cards ──────────────────────────────────────────
            tables = parse_create_tables(sql)
            if tables:
                st.markdown(
                    "<p style='color:#64748b;font-size:12px;'>"
                    "🔑 Primary Key &nbsp; 🔗 Foreign Key &nbsp; ● Not Null</p>",
                    unsafe_allow_html=True,
                )
                render_schema_cards(tables)
            else:
                st.warning("Could not parse structured tables — showing raw output.")

            # ── Raw SQL (always shown, collapsible) ──────────────────────────
            with st.expander("View raw SQL", expanded=(not tables)):
                st.code(sql, language="sql")
