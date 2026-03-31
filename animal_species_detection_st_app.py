"""
Animal Classifier — MobileNet & ShuffleNet
Streamlit app for 10-class animal image classification.
Dataset folders are in Italian; predictions are translated to English for display.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATHS = {
    "MobileNet":  "mobilenet_animals_1_2.pth",
    "ShuffleNet": "shufflenet_animals_1_2.pth",
}

# Alphabetical order of Italian folder names — matches ImageFolder class index assignment
CLASS_NAMES = [
    "cane",       # 0 → dog
    "cavallo",    # 1 → horse
    "elefante",   # 2 → elephant
    "farfalla",   # 3 → butterfly
    "gallina",    # 4 → chicken
    "gatto",      # 5 → cat
    "mucca",      # 6 → cow
    "pecora",     # 7 → sheep
    "ragno",      # 8 → spider
    "scoiattolo", # 9 → squirrel
]

# Italian → English display labels
TRANSLATE = {
    "cane":       "dog",
    "cavallo":    "horse",
    "elefante":   "elephant",
    "farfalla":   "butterfly",
    "gallina":    "chicken",
    "gatto":      "cat",
    "mucca":      "cow",
    "pecora":     "sheep",
    "ragno":      "spider",
    "scoiattolo": "squirrel",
}

ANIMAL_EMOJI = {
    "dog": "🐶", "horse": "🐴", "elephant": "🐘", "butterfly": "🦋",
    "chicken": "🐔", "cat": "🐱", "cow": "🐄", "sheep": "🐑",
    "spider": "🕷️", "squirrel": "🐿️",
}

IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = len(CLASS_NAMES)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def _build_mobilenet() -> nn.Module:
    m = mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
    return m


def _build_shufflenet() -> nn.Module:
    m = shufflenet_v2_x1_0(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> nn.Module:
    """Load and cache a saved model. Runs only once per session."""
    path = MODEL_PATHS[model_name]

    if model_name == "MobileNet":
        model = _build_mobilenet()
    elif model_name == "ShuffleNet":
        model = _build_shufflenet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Handle state_dict, raw dict, or full model saves
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        try:
            model.load_state_dict(checkpoint)
        except Exception:
            model = checkpoint
    else:
        model = checkpoint  # full model saved directly

    model.eval()
    return model


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def prepare_input(image: Image.Image) -> torch.Tensor:
    """PIL image → normalised tensor batch of shape (1, 3, H, W)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return _transform(image).unsqueeze(0)


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def predict_image(model: nn.Module, image: Image.Image, top_k: int = 3):
    """
    Run inference and return top-k predictions as (english_label, confidence_pct) tuples.
    """
    tensor = prepare_input(image)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    top_k = min(top_k, NUM_CLASSES)
    values, indices = torch.topk(probs, top_k)

    results = []
    for val, idx in zip(values, indices):
        italian = CLASS_NAMES[idx.item()]
        english = TRANSLATE.get(italian, italian)  # fallback: show Italian if missing
        results.append((english, round(val.item() * 100, 2)))

    return results


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def _confidence_bar(label: str, pct: float, rank: int):
    emoji = ANIMAL_EMOJI.get(label, "🐾")
    color = "#4CAF50" if rank == 0 else "#90CAF9" if rank == 1 else "#CFD8DC"
    bold  = "700" if rank == 0 else "400"
    size  = "1.05rem" if rank == 0 else "0.95rem"
    st.markdown(
        f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;
                      font-weight:{bold};font-size:{size};">
            <span>{emoji} {label.capitalize()}</span>
            <span>{pct:.1f}%</span>
          </div>
          <div style="background:#e0e0e0;border-radius:999px;height:10px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};
                        border-radius:999px;transition:width .5s ease;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Animal Classifier",
        page_icon="🐾",
        layout="centered",
    )

    # ── Custom CSS ──────────────────────────────
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@400;500&display=swap');

          html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
          h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

          .hero-title {
            font-family: 'Syne', sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -1px;
            line-height: 1.15;
            margin-bottom: 0.2rem;
          }
          .hero-sub {
            color: #607d8b;
            font-size: 1rem;
            margin-bottom: 1.5rem;
          }
          .result-card {
            background: linear-gradient(135deg, #f0fdf4, #e8f5e9);
            border: 1.5px solid #a5d6a7;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
          }
          .model-badge {
            display: inline-block;
            background: #263238;
            color: #ffffff;
            border-radius: 999px;
            font-size: 0.75rem;
            padding: 2px 12px;
            font-family: 'Syne', sans-serif;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 1rem;
          }
          .stButton > button {
            background: #1b5e20;
            color: white;
            border: none;
            border-radius: 10px;
            font-family: 'Syne', sans-serif;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.55rem 1.8rem;
            width: 100%;
            transition: background .2s;
          }
          .stButton > button:hover { background: #2e7d32; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ───────────────────────────────────
    st.markdown('<div class="hero-title">🐾 Animal Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Upload any photo — MobileNet or ShuffleNet will identify the animal.</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Model Settings")
        model_name = st.radio(
            "Choose a model",
            list(MODEL_PATHS.keys()),
            help="Both models are trained on 10 animal classes.",
        )
        st.markdown("---")
        st.markdown("**Supported animals**")
        for english in TRANSLATE.values():
            emoji = ANIMAL_EMOJI.get(english, "🐾")
            st.markdown(f"{emoji} {english.capitalize()}")
        st.markdown("---")
        st.caption("MobileNetV2 & ShuffleNetV2 · Fine-tuned on Animals-10")

    # ── Image uploader ────────────────────────────
    uploaded = st.file_uploader(
        "Drop or select an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("👆 Upload an animal photo to get started.", icon="🖼️")
        return

    image = Image.open(uploaded)

    col_img, col_btn = st.columns([3, 1], vertical_alignment="bottom")
    with col_img:
        st.image(image, caption=uploaded.name, use_container_width=True)
    with col_btn:
        predict_clicked = st.button("🔍 Predict")

    if not predict_clicked:
        return

    # ── Load model ───────────────────────────────
    try:
        with st.spinner(f"Loading {model_name}…"):
            model = load_model(model_name)
    except Exception as e:
        st.error(f"❌ Could not load **{model_name}**: `{e}`")
        st.caption("Make sure the `.pth` files are in the same folder as `app.py`.")
        return

    # ── Run inference ────────────────────────────
    with st.spinner("Running inference…"):
        try:
            top3 = predict_image(model, image, top_k=3)
        except Exception as e:
            st.error(f"❌ Prediction failed: `{e}`")
            return

    # ── Show results ─────────────────────────────
    best_label, best_conf = top3[0]
    best_emoji = ANIMAL_EMOJI.get(best_label, "🐾")

    st.markdown(
        f"""
        <div class="result-card">
          <div class="model-badge">{model_name}</div><br>
          <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;">
            {best_emoji} {best_label.upper()}
          </div>
          <div style="color:#388e3c;font-size:1.05rem;font-weight:500;">
            {best_conf:.1f}% confidence
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Top-3 predictions")
    for rank, (label, pct) in enumerate(top3):
        _confidence_bar(label, pct, rank)


if __name__ == "__main__":
    main()