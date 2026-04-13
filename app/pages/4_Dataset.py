"""Dataset Upload Page - modality-aware"""

import io
import os
import re
import tempfile
import zipfile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset Upload", page_icon="🧠", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 4 of 5")
st.sidebar.progress(4 / 5)
st.sidebar.markdown(
    """
**Steps**
1. Task
2. Agents
3. Instructions
4. Dataset
5. Review
"""
)

st.title("Dataset Upload")
st.caption("Upload your dataset. The upload format adapts to the modalities you selected in Step 1.")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
SHEET_EXTENSIONS = {".csv", ".xlsx"}
N_SAMPLE_IMAGES = 3  # max images to render in preview


# ---------- state ----------
def init_dataset_state():
    defaults = {
        "dataset_df": None,
        "dataset_filename": None,
        "dataset_image_bytes": {},   # {filename_stem: bytes} - only sample images stored
        "dataset_all_image_names": [],  # full list of image filenames found in ZIP
        "dataset_image_dir": None,
        "dataset_image_lookup": {},
        "column_mapping": {},
        "dataset_ready": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_modalities() -> list[str]:
    return st.session_state.get("modalities", ["Image", "Text"])


def has_image(modalities: list[str]) -> bool:
    return "Image" in modalities


def has_text(modalities: list[str]) -> bool:
    return any(m in modalities for m in ["Text", "Tabular / structured data"])


init_dataset_state()

modalities = get_modalities()
mode_image = has_image(modalities)
mode_text = has_text(modalities)


# ---------- helpers ----------

def load_spreadsheet_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    return pd.read_excel(io.BytesIO(raw))


def extract_zip(raw_bytes: bytes) -> tuple:
    """
    Extract a ZIP file.
    Writes all images to a temp directory so they can be read at run time.
    Returns (df, sample_images, all_image_names, image_dir, image_lookup, error_string_or_None)
    """
    df = None
    sample_images: dict[str, bytes] = {}
    all_image_names: list[str] = []
    image_dir: str | None = None
    image_lookup: dict[str, str] = {}

    try:
        tmp_dir = tempfile.mkdtemp(prefix="ai_platform_imgs_")

        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            names = zf.namelist()

            # Find spreadsheet - prefer metadata/ subfolder, fall back to anywhere
            sheet_candidates = [
                n for n in names
                if os.path.splitext(n)[1].lower() in SHEET_EXTENSIONS
                and not os.path.basename(n).startswith(".")
            ]
            metadata_sheets = [n for n in sheet_candidates if "metadata" in n.lower()]
            chosen_sheet = (metadata_sheets or sheet_candidates or [None])[0]

            if chosen_sheet:
                sheet_filename = os.path.basename(chosen_sheet)
                df = load_spreadsheet_bytes(zf.read(chosen_sheet), sheet_filename)

            # Find images
            image_files = [
                n for n in names
                if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
                and not os.path.basename(n).startswith(".")
            ]
            all_image_names = [os.path.basename(n) for n in image_files]

            # Write all images to temp dir; keep sample bytes for preview
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
 
        image_dir = tmp_dir
 
    except zipfile.BadZipFile:
        return None, {}, [], None, {}, "The uploaded file is not a valid ZIP archive."
    except Exception as e:
        return None, {}, [], None, {}, str(e)
 
    return df, sample_images, all_image_names, image_dir, image_lookup, None


def persist_dataset(df, sample_images, all_image_names, image_dir, image_lookup, filename):
    st.session_state.dataset_df = df
    st.session_state.dataset_filename = filename
    st.session_state.dataset_image_bytes = sample_images
    st.session_state.dataset_all_image_names = all_image_names
    st.session_state.dataset_image_dir = image_dir  # None for text-only datasets
    st.session_state.dataset_image_lookup = image_lookup or {}
    st.session_state.dataset_ready = True

    cfg = st.session_state.setdefault("experiment_config", {})
    cfg["dataset"] = {
        "filename": filename,
        "num_rows": len(df) if df is not None else 0,
        "columns": list(df.columns) if df is not None else [],
        "num_images": len(all_image_names),
    }


# ---------- main layout ----------
main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:

    # ── Upload section ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("1. Upload dataset")

        if mode_image and mode_text:
            st.info(
                "**Expected format: ZIP file**  \n"
                "Put your images at the root of the ZIP and your spreadsheet "
                "(CSV or Excel) inside a `metadata/` subfolder.  \n"
                "Image filenames should match the ID column - e.g. `ad_001.jpg` ↔ `ad_001`.",
                icon="📦",
            )
            uploader_label = "Upload ZIP file (images + metadata spreadsheet)"
            uploader_types = ["zip"]

        elif mode_image and not mode_text:
            st.info(
                "**Expected format: ZIP file**  \n"
                "Put all your images inside the ZIP. No spreadsheet required.",
                icon="🖼️",
            )
            uploader_label = "Upload ZIP file (images only)"
            uploader_types = ["zip"]

        elif mode_text and not mode_image:
            st.info(
                "**Expected format: CSV or Excel file**  \n"
                "One row per item.",
                icon="📄",
            )
            uploader_label = "Upload CSV or Excel file"
            uploader_types = ["csv", "xlsx"]

        else:
            st.info("Upload a ZIP or spreadsheet containing your dataset.", icon="📁")
            uploader_label = "Upload dataset file"
            uploader_types = ["zip", "csv", "xlsx"]

        uploaded = st.file_uploader(uploader_label, type=uploader_types, key="dataset_uploader")

        if uploaded is not None:
            if st.button("Load dataset", type="primary"):
                raw = uploaded.read()
                fname = uploaded.name

                with st.spinner("Reading dataset…"):
                    if fname.endswith(".zip"):
                        df, sample_imgs, all_imgs, image_dir, image_lookup, err = extract_zip(raw)
                        if err:
                            st.error(f"Failed to load ZIP: {err}")
                        else:
                            persist_dataset(df, sample_imgs, all_imgs, image_dir, image_lookup, fname)
                            n_imgs = len(all_imgs)
                            n_rows = len(df) if df is not None else 0
                            msg = f"Loaded {n_imgs} image(s)"
                            if df is not None:
                                msg += f" and metadata with {n_rows} rows."
                            st.success(msg)
                            st.rerun()
                    else:
                        try:
                            df = load_spreadsheet_bytes(raw, fname)
                            persist_dataset(df, {}, [], None, {}, fname)
                            st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load file: {e}")

    # ── Preview sections (only after successful load) ─────────────────────────
    if st.session_state.dataset_ready:
        df = st.session_state.dataset_df
        sample_imgs = st.session_state.dataset_image_bytes
        all_imgs = st.session_state.dataset_all_image_names
        section = 2  # rolling section counter

        # Sample images
        if sample_imgs:
            with st.container(border=True):
                st.subheader(f"{section}. Sample images")
                section += 1
                st.caption(
                    f"Showing {len(sample_imgs)} of {len(all_imgs)} image(s). "
                    "Only a small sample is loaded here - the full set is read at run time."
                )
                img_cols = st.columns(len(sample_imgs))
                for col, (stem, img_bytes) in zip(img_cols, sample_imgs.items()):
                    with col:
                        st.image(img_bytes, caption=stem, width="stretch")

        # Metadata table
        if df is not None:
            with st.container(border=True):
                st.subheader(f"{section}. Metadata preview")
                section += 1
                st.caption(
                    f"**{st.session_state.dataset_filename}** - "
                    f"{len(df)} rows × {len(df.columns)} columns"
                )
                all_cols = list(df.columns)
                selected_cols = st.multiselect(
                    "Columns to display",
                    options=all_cols,
                    default=all_cols[:min(6, len(all_cols))],
                    key="preview_cols",
                )
                n_preview = st.slider(
                    "Rows to preview",
                    min_value=5,
                    max_value=min(50, len(df)),
                    value=10,
                    step=5,
                )
                if selected_cols:
                    st.dataframe(df[selected_cols].head(n_preview), width='stretch')
                else:
                    st.info("Select at least one column to display.")

        # Column mapping
        if df is not None:
            with st.container(border=True):
                st.subheader(f"{section}. Column mapping")
                st.caption(
                    "Map spreadsheet columns to the input fields you defined in Step 1. "
                    "For image fields, select the ID column whose values match image filenames."
                )

                input_fields = st.session_state.get("input_fields", [])
                if not input_fields:
                    st.warning("No input fields defined yet. Go back to Step 1 to set up your input schema.")
                else:
                    col_options = ["- not mapped -"] + list(df.columns)
                    mapping = st.session_state.column_mapping

                    for field in input_fields:
                        fname = field["name"]
                        ftype = field["type"]
                        current = mapping.get(fname, "- not mapped -")
                        idx = col_options.index(current) if current in col_options else 0

                        if ftype == "Image":
                            label = f"`{fname}` (Image) - ID column used to match image filenames"
                        else:
                            label = f"`{fname}` ({ftype})"

                        mapping[fname] = st.selectbox(label, col_options, index=idx, key=f"col_map_{fname}")

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
                st.warning("Please upload a dataset before continuing.")
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

            st.markdown(f"**File**  \n`{st.session_state.dataset_filename}`")
            if all_imgs:
                st.markdown(f"**Images**  \n{len(all_imgs)}")
            if df is not None:
                st.markdown(f"**Rows**  \n{len(df)}")
                st.markdown(f"**Columns**  \n{len(df.columns)}")
                st.divider()
                st.markdown("**Columns**")
                for col in df.columns:
                    st.markdown(f"- `{col}`")

            mapping = {k: v for k, v in st.session_state.column_mapping.items() if v != "- not mapped -"}
            if mapping:
                st.divider()
                st.markdown("**Column mapping**")
                for field, col in mapping.items():
                    st.markdown(f"- `{field}` ← `{col}`")
        else:
            st.markdown("No dataset loaded yet.")

    with st.container(border=True):
        st.subheader("Tips")
        st.markdown(
            """
- ZIP: images at root, spreadsheet in `metadata/`
- Filenames should match the ID column (no extension)
- Only 3 sample images load in preview - full set loads at run time
- You can re-upload to replace the current dataset
"""
        )
