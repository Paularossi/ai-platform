"""Step 4 — Dataset. Supports inline entry, file upload (CSV/Excel), and ZIP (images + metadata)."""

import io
import os
import re
import tempfile
import zipfile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset", page_icon="🧠", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 4 of 5")
st.sidebar.progress(4 / 5)
st.sidebar.markdown("""
**Steps**
1. Task
2. Agents
3. Instructions
4. Dataset
5. Review
""")

st.title("Dataset")
st.caption("Provide the items your agents will deliberate on.")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
SHEET_EXTENSIONS = {".csv", ".xlsx"}
N_SAMPLE_IMAGES = 3


# ---------- state ----------
def init_state():
    defaults = {
        "dataset_df": None,
        "dataset_filename": None,
        "dataset_image_bytes": {},
        "dataset_all_image_names": [],
        "dataset_image_dir": None,
        "dataset_image_lookup": {},
        "column_mapping": {},
        "dataset_ready": False,
        "dataset_source": "inline",   # "inline" | "file"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_modalities() -> list[str]:
    return st.session_state.get("modalities", ["Text"])


def has_image(modalities: list[str]) -> bool:
    return "Image" in modalities


init_state()
modalities = get_modalities()
mode_image = has_image(modalities)
input_fields = st.session_state.get("input_fields", [])


# ---------- helpers ----------
def load_spreadsheet(raw: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    return pd.read_excel(io.BytesIO(raw))


def extract_zip(raw_bytes: bytes):
    """Returns (df, sample_images, all_image_names, image_dir, image_lookup, error)"""
    df = None
    sample_images: dict[str, bytes] = {}
    all_image_names: list[str] = []
    image_lookup: dict[str, str] = {}

    try:
        tmp_dir = tempfile.mkdtemp(prefix="ai_platform_imgs_")
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            names = zf.namelist()

            # Find spreadsheet — prefer metadata/ subfolder
            sheets = [n for n in names if os.path.splitext(n)[1].lower() in SHEET_EXTENSIONS
                      and not os.path.basename(n).startswith(".")]
            meta_sheets = [n for n in sheets if "metadata" in n.lower()]
            chosen = (meta_sheets or sheets or [None])[0]
            if chosen:
                df = load_spreadsheet(zf.read(chosen), os.path.basename(chosen))

            # Find images
            image_files = [n for n in names if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
                           and not os.path.basename(n).startswith(".")]
            all_image_names = [os.path.basename(n) for n in image_files]

            for img_path in image_files:
                img_bytes = zf.read(img_path)
                basename = os.path.basename(img_path)
                stem = os.path.splitext(basename)[0]
                full_path = os.path.join(tmp_dir, basename)
                with open(full_path, "wb") as f:
                    f.write(img_bytes)
                norm = re.sub(r"\.0+$", "", stem.strip().lower())
                image_lookup[norm] = full_path
                if len(sample_images) < N_SAMPLE_IMAGES:
                    sample_images[stem] = img_bytes

    except zipfile.BadZipFile:
        return None, {}, [], None, {}, "Not a valid ZIP archive."
    except Exception as e:
        return None, {}, [], None, {}, str(e)

    return df, sample_images, all_image_names, tmp_dir, image_lookup, None


def persist_dataset(df, sample_images, all_image_names, image_dir, image_lookup, filename):
    st.session_state.dataset_df = df
    st.session_state.dataset_filename = filename
    st.session_state.dataset_image_bytes = sample_images
    st.session_state.dataset_all_image_names = all_image_names
    st.session_state.dataset_image_dir = image_dir
    st.session_state.dataset_image_lookup = image_lookup or {}
    st.session_state.dataset_ready = True
    cfg = st.session_state.setdefault("experiment_config", {})
    cfg["dataset"] = {
        "filename": filename,
        "num_rows": len(df) if df is not None else 0,
        "columns": list(df.columns) if df is not None else [],
        "num_images": len(all_image_names),
        "source": st.session_state.dataset_source,
    }


# ---------- layout ----------
main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:

    # ── Source selector ───────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("1. How do you want to provide the data?")

        source = st.radio(
            "Dataset source",
            ["inline", "file"],
            format_func=lambda s: {
                "inline": "Type items directly (small experiments)",
                "file": "Upload a file (CSV, Excel, or ZIP with images)",
            }[s],
            key="dataset_source",
            horizontal=True,
        )

    # ── Inline entry ──────────────────────────────────────────────────────────
    if source == "inline":
        with st.container(border=True):
            st.subheader("2. Enter items")

            if not input_fields:
                st.warning("No input fields defined. Go back to Step 1 to set up your input schema.")
            else:
                st.caption(
                    "Fill in one row per item. Add rows with the button below. "
                    "These values will be passed directly to agents as item data."
                )

                # Initialise inline rows if not present
                if "inline_rows" not in st.session_state:
                    st.session_state.inline_rows = [
                        {f["name"]: "" for f in input_fields}
                    ]

                rows = st.session_state.inline_rows
                field_names = [f["name"] for f in input_fields]

                for i, row in enumerate(rows):
                    cols = st.columns([*[3] * len(field_names), 0.5])
                    for j, fname in enumerate(field_names):
                        with cols[j]:
                            row[fname] = st.text_input(
                                fname if i == 0 else "",
                                value=row.get(fname, ""),
                                key=f"inline_{i}_{fname}",
                                label_visibility="visible" if i == 0 else "collapsed",
                            )
                    with cols[-1]:
                        if i == 0:
                            st.write("")
                        if st.button("✕", key=f"rm_inline_{i}", disabled=len(rows) == 1):
                            rows.pop(i)
                            st.rerun()

                if st.button("+ Add row"):
                    rows.append({f["name"]: "" for f in input_fields})
                    st.rerun()

                st.divider()

                if st.button("Use these items", type="primary"):
                    filled = [r for r in rows if any(v.strip() for v in r.values())]
                    if not filled:
                        st.warning("Please fill in at least one item.")
                    else:
                        df = pd.DataFrame(filled)
                        persist_dataset(df, {}, [], None, {}, "inline")
                        st.success(f"Loaded {len(df)} item(s) from inline entry.")
                        st.rerun()

    # ── File upload ───────────────────────────────────────────────────────────
    else:
        with st.container(border=True):
            st.subheader("2. Upload file")

            if mode_image:
                st.info(
                    "**ZIP file** — images at the root, spreadsheet (CSV or Excel) "
                    "in a `metadata/` subfolder. Image filenames must match the ID "
                    "column stem (e.g. `item_001.jpg` ↔ `item_001`).",
                    icon="📦",
                )
                uploader_label = "Upload ZIP file"
                uploader_types = ["zip", "csv", "xlsx"]
            else:
                st.info(
                    "**CSV or Excel** — one row per item, one column per input field.",
                    icon="📄",
                )
                uploader_label = "Upload CSV or Excel file"
                uploader_types = ["csv", "xlsx"]

            uploaded = st.file_uploader(uploader_label, type=uploader_types, key="dataset_uploader")

            if uploaded is not None:
                if st.button("Load dataset", type="primary"):
                    raw = uploaded.read()
                    fname = uploaded.name
                    with st.spinner("Reading…"):
                        if fname.endswith(".zip"):
                            df, samps, all_imgs, img_dir, lookup, err = extract_zip(raw)
                            if err:
                                st.error(f"Failed to load ZIP: {err}")
                            else:
                                persist_dataset(df, samps, all_imgs, img_dir, lookup, fname)
                                n_imgs = len(all_imgs)
                                n_rows = len(df) if df is not None else 0
                                msg = f"Loaded {n_imgs} image(s)"
                                if df is not None:
                                    msg += f" and metadata with {n_rows} rows."
                                st.success(msg)
                                st.rerun()
                        else:
                            try:
                                df = load_spreadsheet(raw, fname)
                                persist_dataset(df, {}, [], None, {}, fname)
                                st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to load file: {e}")

    # ── Preview (shown after any successful load) ─────────────────────────────
    if st.session_state.dataset_ready:
        df = st.session_state.dataset_df
        sample_imgs = st.session_state.dataset_image_bytes
        all_imgs = st.session_state.dataset_all_image_names
        section = 3

        if sample_imgs:
            with st.container(border=True):
                st.subheader(f"{section}. Sample images")
                section += 1
                st.caption(
                    f"Showing {len(sample_imgs)} of {len(all_imgs)} image(s). "
                    "Full set loads at run time."
                )
                img_cols = st.columns(len(sample_imgs))
                for col, (stem, img_bytes) in zip(img_cols, sample_imgs.items()):
                    with col:
                        st.image(img_bytes, caption=stem, use_container_width=True)

        if df is not None:
            with st.container(border=True):
                st.subheader(f"{section}. Data preview")
                section += 1
                st.caption(
                    f"**{st.session_state.dataset_filename}** — "
                    f"{len(df)} row(s) × {len(df.columns)} column(s)"
                )
                all_cols = list(df.columns)
                sel_cols = st.multiselect(
                    "Columns to show",
                    all_cols,
                    default=all_cols[:min(6, len(all_cols))],
                    key="preview_cols",
                )
                if len(df) > 1:
                    n_preview = st.slider("Rows to preview", 1, min(20, len(df)), min(5, len(df)), key="preview_rows")
                else:
                    n_preview = 1
                    st.caption("1 row — showing all.")
                if sel_cols:
                    st.dataframe(df[sel_cols].head(n_preview), width="stretch")
                else:
                    st.info("Select at least one column to display.")

        # Column mapping — only needed for file uploads; inline maps automatically
        if df is not None and st.session_state.dataset_source == "file":
            with st.container(border=True):
                st.subheader(f"{section}. Column mapping")
                st.caption(
                    "Map spreadsheet columns to the input fields from Step 1. "
                    "For image tasks, pick the ID column whose values match image filenames."
                )
                if not input_fields:
                    st.warning("No input fields defined. Go back to Step 1.")
                else:
                    col_options = ["— not mapped —"] + list(df.columns)
                    mapping = st.session_state.column_mapping
                    for field in input_fields:
                        fname = field["name"]
                        ftype = field["type"]
                        current = mapping.get(fname, "— not mapped —")
                        idx = col_options.index(current) if current in col_options else 0
                        label = (
                            f"`{fname}` (Image) — ID column for matching filenames"
                            if ftype == "Image" else f"`{fname}` ({ftype})"
                        )
                        mapping[fname] = st.selectbox(label, col_options, index=idx, key=f"col_map_{fname}")
                    st.session_state.column_mapping = mapping
                    st.session_state.experiment_config.setdefault("dataset", {})["column_mapping"] = mapping
        elif df is not None and st.session_state.dataset_source == "inline":
            # Inline: column names already match field names — auto-map
            mapping = {f["name"]: f["name"] for f in input_fields if f["name"] in df.columns}
            st.session_state.column_mapping = mapping
            st.session_state.experiment_config.setdefault("dataset", {})["column_mapping"] = mapping

    # ── Navigation ─────────────────────────────────────────────────────────────
    st.divider()
    nav1, nav2, nav3 = st.columns([2, 2, 0.8])
    with nav1:
        if st.button("← Back"):
            st.switch_page("pages/3_Instructions.py")
    with nav2:
        if st.button("Save draft"):
            st.success("Dataset setup saved.")
    with nav3:
        if st.button("Next →"):
            if not st.session_state.dataset_ready:
                st.warning("Please provide a dataset before continuing.")
            else:
                st.switch_page("pages/5_Review.py")


# ---------- summary col ──────────────────────────────────────────────────────
with summary_col:
    with st.container(border=True):
        st.subheader("Live summary")
        st.markdown(f"**Modalities**  \n{', '.join(modalities)}")
        st.divider()

        if st.session_state.dataset_ready:
            df = st.session_state.dataset_df
            all_imgs = st.session_state.dataset_all_image_names
            source_label = "Inline" if st.session_state.dataset_source == "inline" else st.session_state.dataset_filename
            st.markdown(f"**Source**  \n{source_label}")
            if all_imgs:
                st.markdown(f"**Images**  \n{len(all_imgs)}")
            if df is not None:
                st.markdown(f"**Rows**  \n{len(df)}")
                st.markdown(f"**Columns**  \n{', '.join(df.columns)}")
            mapping = {k: v for k, v in st.session_state.column_mapping.items() if v != "— not mapped —"}
            if mapping:
                st.divider()
                st.markdown("**Column mapping**")
                for field, col in mapping.items():
                    st.markdown(f"- `{field}` ← `{col}`")
        else:
            st.markdown("No data provided yet.")

    with st.container(border=True):
        st.subheader("Tips")
        st.markdown(
            """
- **Inline**: best for quick tests with a few items — no file needed
- **File**: CSV/Excel for text tasks; ZIP for image + text tasks
- ZIP structure: images at root, spreadsheet in `metadata/`
- Image filenames must match the ID column (stem only, no extension)
- Inline items auto-map to your input fields from Step 1
"""
        )