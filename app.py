"""
app.py — Deteksi Penyakit Daun Stroberi (Foggy / Low-Light)
=============================================================
Web app Streamlit untuk deployment model klasifikasi penyakit daun stroberi.

Dua model dimuat sekaligus:
  Model A (Baseline) : model_terbaik_model_A_baseline.keras  → resize + normalize
  Model B (Pipeline) : model_terbaik_model_B_pipeline.keras  → pipeline lengkap

Cara pakai:
    1. Taruh kedua model di folder model/
    2. pip install -r requirements.txt
    3. python -m streamlit run app.py
"""

import base64
import io
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from preprocessing import run_dual_inference

MODEL_A_PATH  = os.path.join("model", "model_terbaik_model_A_baseline.keras")
MODEL_B_PATH  = os.path.join("model", "model_terbaik_model_B_pipeline.keras")
METADATA_PATH = os.path.join("model", "metadata_model.json")

st.set_page_config(
    page_title="Diagnosa Daun Stroberi",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Work+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --fog:            #F3F6F1;
            --card:           #FFFDF9;
            --ink:            #1E3428;
            --strawberry:     #C1392B;
            --strawberry-tint:#F7E3E0;
            --leaf:           #4F7942;
            --leaf-tint:      #E6EFE1;
            --umber:          #7A6652;
            --umber-line:     #DDD6C8;
            --purple-tint:    #EAE6FF;
            --purple:         #4A3F8C;
        }

        html, body, [class*="css"] { font-family:'Work Sans',sans-serif; color:var(--ink); }
        .stApp { background:var(--fog); }
        #MainMenu, header, footer { visibility:hidden; }
        .block-container { padding-top:1.5rem !important; padding-bottom:1.5rem !important; max-width:1100px !important; }

        /* ── Hero ── */
        .sp-hero { margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid var(--umber-line); }
        .sp-eyebrow {
            font-family:'JetBrains Mono',monospace; font-size:0.68rem;
            letter-spacing:0.14em; text-transform:uppercase; color:var(--umber);
            margin-bottom:0.2rem;
        }
        .sp-title {
            font-family:'Fraunces',serif; font-weight:600;
            font-size:2.2rem; line-height:1.1; margin:0; color:var(--ink);
        }
        .sp-title em { color:var(--strawberry); font-style:normal; }
        .sp-title strong { color:var(--leaf); font-weight:inherit; }
        .sp-tagline { font-size:0.88rem; color:#46594E; margin-top:0.25rem; line-height:1.5; }

        /* ── Model pills ── */
        .sp-pills { display:flex; gap:0.6rem; margin-top:0.7rem; flex-wrap:wrap; }
        .sp-pill {
            display:flex; align-items:center; gap:0.5rem;
            padding:0.35rem 0.75rem; border-radius:6px;
            font-size:0.78rem; line-height:1.4; flex:1; min-width:200px;
        }
        .sp-pill.a { background:var(--purple-tint); color:var(--purple); }
        .sp-pill.b { background:var(--leaf-tint);   color:var(--leaf); }
        .sp-pill-badge {
            font-family:'JetBrains Mono',monospace; font-size:0.65rem;
            font-weight:600; letter-spacing:0.06em; text-transform:uppercase;
            white-space:nowrap; flex-shrink:0;
        }

        /* ── Section label ── */
        .sp-label {
            font-family:'JetBrains Mono',monospace; font-size:0.65rem;
            letter-spacing:0.12em; text-transform:uppercase; color:var(--umber);
            margin-bottom:0.4rem;
        }

        /* ── Upload zone ── */
        [data-testid="stFileUploader"] { height:100%; }
        [data-testid="stFileUploader"] section {
            background:var(--card) !important;
            border:2px dashed var(--umber-line) !important;
            border-radius:10px !important;
            padding:0 !important;
            min-height:320px;
            display:flex !important;
            flex-direction:column !important;
            align-items:center !important;
            justify-content:center !important;
            transition:border-color 0.25s, background 0.25s;
            cursor:pointer;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color:var(--leaf) !important;
            background:var(--leaf-tint) !important;
        }
        [data-testid="stFileUploader"] section > div {
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            gap:0.4rem; padding:2.5rem 2rem; text-align:center; width:100%;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] svg {
            width:40px !important; height:40px !important;
            color:var(--umber) !important; margin-bottom:0.3rem;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            font-family:'Work Sans',sans-serif !important;
            font-size:0.9rem !important; color:var(--ink) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            font-family:'JetBrains Mono',monospace !important;
            font-size:0.68rem !important; color:var(--umber) !important;
        }
        [data-testid="stFileUploader"] section button {
            background:var(--ink) !important; color:var(--fog) !important;
            border:none !important; border-radius:5px !important;
            font-family:'JetBrains Mono',monospace !important;
            font-size:0.72rem !important; letter-spacing:0.06em !important;
            padding:0.45rem 1.2rem !important; margin-top:0.6rem !important;
            cursor:pointer !important;
        }
        [data-testid="stFileUploader"] section button:hover { opacity:0.85 !important; }

        /* ── Preview frame ── */
        .sp-img-frame {
            border-radius:10px; overflow:hidden;
            border:1px solid var(--umber-line); background:var(--card);
            min-height:320px; display:flex; align-items:center;
        }
        .sp-img-frame img { width:100%; display:block; object-fit:cover; }

        /* ── Diagnosis card ── */
        .sp-dcard {
            background:var(--fog); border:1px solid var(--umber-line);
            border-radius:10px; overflow:hidden;
        }
        .sp-dcard.winner { border-color:var(--leaf); box-shadow:0 0 0 2.5px var(--leaf-tint); }

        /* badge row at top */
        .sp-dcard-head {
            padding:0.7rem 0.9rem 0.5rem;
            display:flex; align-items:center; gap:0.5rem;
        }
        .sp-model-badge {
            font-family:'JetBrains Mono',monospace; font-size:0.63rem;
            letter-spacing:0.1em; text-transform:uppercase;
            padding:0.15rem 0.5rem; border-radius:3px;
            display:inline-block; white-space:nowrap;
        }
        .sp-model-badge.a { background:var(--purple-tint); color:var(--purple); }
        .sp-model-badge.b { background:var(--leaf-tint);   color:var(--leaf); }

        /* image with corner brackets */
        .sp-bracket-wrap {
            position:relative; margin:0 0.9rem 0; background:var(--card);
            border-radius:6px; overflow:hidden; border:1px solid var(--umber-line);
        }
        .sp-bracket-wrap img { width:100%; display:block; max-height:240px; object-fit:cover; }
        .corner {
            position:absolute; width:14px; height:14px;
            border-color:var(--leaf); border-style:solid; z-index:2;
        }
        .corner.tl { top:5px; left:5px;  border-width:2px 0 0 2px; }
        .corner.tr { top:5px; right:5px; border-width:2px 2px 0 0; }
        .corner.bl { bottom:5px; left:5px;  border-width:0 0 2px 2px; }
        .corner.br { bottom:5px; right:5px; border-width:0 2px 2px 0; }

        /* bottom verdict row */
        .sp-dcard-foot {
            padding:0.7rem 0.9rem 0.8rem;
            display:flex; justify-content:space-between; align-items:flex-end;
        }
        .sp-verdict {
            font-family:'Fraunces',serif; font-weight:700;
            font-size:1.7rem; letter-spacing:0.02em; line-height:1;
        }
        .sp-verdict.sehat { color:var(--leaf); }
        .sp-verdict.sakit { color:var(--strawberry); }
        .sp-foot-right { text-align:right; }
        .sp-conf-pct {
            font-family:'JetBrains Mono',monospace;
            font-size:1.4rem; font-weight:600; line-height:1;
        }
        .sp-conf-label {
            font-family:'JetBrains Mono',monospace; font-size:0.6rem;
            color:var(--umber); text-transform:uppercase; letter-spacing:0.08em;
            margin-top:0.15rem;
        }
        .sp-bar-track { background:var(--umber-line); border-radius:3px; height:4px; overflow:hidden; margin:0 0.9rem 0.9rem; }
        .sp-bar-fill  { height:100%; border-radius:3px; }

        /* ── Notice ── */
        .sp-notice {
            background:var(--strawberry-tint); border:1px solid var(--strawberry);
            border-radius:8px; padding:1.1rem 1.3rem; font-size:0.9rem; line-height:1.6;
        }
        .sp-notice code {
            background:rgba(0,0,0,0.06); padding:0.1rem 0.35rem;
            border-radius:3px; font-family:'JetBrains Mono',monospace; font-size:0.82rem;
        }

        @media (max-width:640px) {
            .sp-pills { flex-direction:column; }
            .block-container { padding-left:1rem !important; padding-right:1rem !important; }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models_and_meta():
    import tensorflow as tf
    model_a, model_b = None, None
    if os.path.exists(MODEL_A_PATH):
        model_a = tf.keras.models.load_model(MODEL_A_PATH)
    if os.path.exists(MODEL_B_PATH):
        model_b = tf.keras.models.load_model(MODEL_B_PATH)
    meta = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
    return model_a, model_b, meta


def bgr_to_base64(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────
model_a, model_b, meta = load_models_and_meta()

# ─────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sp-hero">
    <div class="sp-eyebrow">🍓 Klasifikasi Daun Stroberi · MobileNetV2</div>
    <h1 class="sp-title">Cek <strong>Daun</strong> Stroberi<br>Sehat, atau <em>sakit</em>?</h1>
    <div class="sp-tagline">Upload foto daun → diagnosis daun dengan Model A &amp; B.</div>
    <div class="sp-pills">
        <div class="sp-pill a">
            <span class="sp-pill-badge">Model A</span>
            Tanpa pre-processing — resize + normalize saja.
        </div>
        <div class="sp-pill b">
            <span class="sp-pill-badge">Model B</span>
            Dengan pipeline lengkap: CLAHE, denoising, dehazing, gamma correction.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Model not found notice
# ─────────────────────────────────────────────────────────────────────
if model_a is None and model_b is None:
    st.markdown(f"""
    <div class="sp-notice">
        <strong>Model belum ditemukan.</strong><br><br>
        Taruh kedua file model di folder <code>model/</code>:<br>
        <code>{MODEL_A_PATH}</code> &nbsp;← Model A (baseline)<br>
        <code>{MODEL_B_PATH}</code> &nbsp;← Model B (pipeline)<br><br>
        (opsional) metadata di <code>{METADATA_PATH}</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if model_a is None:
    st.warning(f"⚠️ Model A tidak ditemukan di `{MODEL_A_PATH}`")
if model_b is None:
    st.warning(f"⚠️ Model B tidak ditemukan di `{MODEL_B_PATH}`")

# ─────────────────────────────────────────────────────────────────────
# Baris 1: Upload | Preview
# ─────────────────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 1], gap="medium")

with col_upload:
    st.markdown('<div class="sp-label">Upload Citra Daun</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload foto daun stroberi",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with col_preview:
    if uploaded is not None:
        file_bytes  = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr     = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        pil_orig    = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img_orig_b64 = pil_to_base64(pil_orig)
        st.markdown('<div class="sp-label">Citra Diunggah</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sp-img-frame"><img src="data:image/png;base64,{img_orig_b64}"></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────
# Baris 2: Diagnosis A | Diagnosis B
# ─────────────────────────────────────────────────────────────────────
if uploaded is not None:
    with st.spinner("Menjalankan dua jalur pre-processing…"):
        result = run_dual_inference(img_bgr, model_a, model_b)

    ra = result["model_a"]
    rb = result["model_b"]
    winner_a = ra["confidence"] > rb["confidence"]

    conf_a_pct = ra["confidence"] * 100
    conf_b_pct = rb["confidence"] * 100
    color_a    = "var(--leaf)" if ra["label"] == "sehat" else "var(--strawberry)"
    color_b    = "var(--leaf)" if rb["label"] == "sehat" else "var(--strawberry)"
    wclass_a   = "winner" if winner_a else ""
    wclass_b   = "winner" if not winner_a else ""

    img_baseline_b64 = bgr_to_base64(result["image_baseline"])
    img_pipeline_b64 = bgr_to_base64(result["image_processed"])

    st.markdown('<div class="sp-label" style="margin-top:0.8rem;">Diagnosis: Model A vs Model B</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1], gap="medium")

    def card_html(badge_cls, badge_txt, img_b64, label, pct, color, wclass):
        return f"""
        <div class="sp-dcard {wclass}">
            <div class="sp-dcard-head">
                <span class="sp-model-badge {badge_cls}">{badge_txt}</span>
            </div>
            <div class="sp-bracket-wrap">
                <span class="corner tl"></span>
                <span class="corner tr"></span>
                <span class="corner bl"></span>
                <span class="corner br"></span>
                <img src="data:image/png;base64,{img_b64}">
            </div>
            <div class="sp-dcard-foot">
                <div class="sp-verdict {label}">{label.upper()}</div>
                <div class="sp-foot-right">
                    <div class="sp-conf-pct" style="color:{color};">{pct:.1f}%</div>
                    <div class="sp-conf-label">Keyakinan</div>
                </div>
            </div>
            <div class="sp-bar-track">
                <div class="sp-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
        </div>
        """

    with col_a:
        st.markdown(card_html(
            "a", "Model A — Baseline",
            img_baseline_b64, ra["label"], conf_a_pct, color_a, wclass_a
        ), unsafe_allow_html=True)

    with col_b:
        st.markdown(card_html(
            "b", "Model B — Pipeline",
            img_pipeline_b64, rb["label"], conf_b_pct, color_b, wclass_b
        ), unsafe_allow_html=True)
