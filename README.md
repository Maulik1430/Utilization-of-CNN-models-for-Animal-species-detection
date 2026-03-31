# 🐾 Animal Classifier — MobileNetV2 & ShuffleNetV2

> A deep learning web app that identifies 10 animal species from photos using transfer learning.  
> Built with PyTorch and deployed via Streamlit — clean, fast, and portfolio-ready.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation-guide)
- [How to Run](#-how-to-run-locally)
- [Deployment](#-deploy-on-streamlit-cloud)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 Overview

This project is a full end-to-end deep learning pipeline — from training models to deploying a live web app. The idea was simple: take two lightweight, mobile-friendly neural networks, fine-tune them on a real-world animal dataset, and wrap everything into a clean Streamlit interface that anyone can use without touching code.

The app takes an uploaded image and predicts which of 10 animal categories it belongs to, showing the top-3 predictions with confidence scores. Two models are available to compare — **MobileNetV2** and **ShuffleNetV2** — both initialized from ImageNet pretrained weights and fine-tuned on the Animals-10 dataset.

This was built as a learning project to understand transfer learning, hyperparameter tuning, and ML deployment end-to-end.

---

## 🌐 Live Demo

> 🚀 Deploy your own version by following the [Deployment section](#-deploy-on-streamlit-cloud) below.

---

## ✨ Features

- 📤 **Drag-and-drop image upload** — supports JPG, PNG, WEBP, BMP
- 🤖 **Two model options** — switch between MobileNetV2 and ShuffleNetV2 in the sidebar
- 📊 **Top-3 predictions** with animated confidence bars
- ⚡ **Model caching** via `@st.cache_resource` — loads once, stays fast
- 🎨 **Clean, styled UI** — custom fonts, color-coded result cards
- 🌍 **Bilingual label handling** — dataset folders are Italian, displayed in English
- 🛡️ **Graceful error handling** — clear messages if model loading or prediction fails

---

## 🛠️ Tech Stack

| Layer | Tools Used |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+, TorchVision |
| **Models** | MobileNetV2, ShuffleNetV2 (pretrained on ImageNet) |
| **Web App** | Streamlit 1.32+ |
| **Image Processing** | Pillow (PIL) |
| **Training Environment** | Google Colab (GPU) |
| **Deployment** | Streamlit Community Cloud |

---

## 🧬 Model Details

Both models follow the same transfer learning strategy: start from ImageNet pretrained weights, replace the final classification layer with a 10-class head, and fine-tune on the Animals-10 dataset.

### Architecture Changes

**MobileNetV2**
```
Original classifier[1]: Linear(1280 → 1000)
Replaced with:          Linear(1280 → 10)
```

**ShuffleNetV2**
```
Original fc: Linear(1024 → 1000)
Replaced with: Linear(1024 → 10)
```

### Training Setup

| Config | Value |
|---|---|
| **Input size** | 224 × 224 × 3 |
| **Normalization** | ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| **Loss function** | CrossEntropyLoss |
| **Optimizers tested** | Adam, SGD (momentum=0.9) |
| **Learning rates tested** | 0.001, 0.0005, 0.0001 |
| **Batch sizes tested** | 16, 32 |
| **Epochs (grid search)** | 2 per combination |

### Hyperparameter Tuning

A full **grid search** was run across all combinations of batch sizes, learning rates, and optimizers — 12 combinations per model, on both raw and augmented data. `itertools.product()` was used to generate all combinations programmatically.

```python
batch_sizes     = [16, 32]
learning_rates  = [0.001, 0.0005, 0.0001]
optimizers_list = ["adam", "sgd"]
# → 12 total combinations per model
```

### Data Augmentation (for augmented training runs)

```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
```

> 💡 **Note:** The final saved models use the best-performing hyperparameter configuration found during grid search.

---

## 📦 Dataset

**Animals-10** by Alessio Corrado — scraped from Google Images and human-verified.

| Property | Details |
|---|---|
| **Source** | [Kaggle — Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) |
| **Total images** | ~26,000 |
| **Download size** | ~583 MB |
| **Classes** | 10 |
| **Image format** | JPEG, varying sizes |
| **Known issue** | Dataset is imbalanced (spider & dog have most images) |

### Class Distribution (approximate)

| Italian Folder | English Label | ~Images |
|---|---|---|
| ragno | 🕷️ spider | 4,800 |
| cane | 🐶 dog | 4,000 |
| cavallo | 🐴 horse | 2,600 |
| gallina | 🐔 chicken | 3,100 |
| farfalla | 🦋 butterfly | 2,100 |
| mucca | 🐄 cow | 1,900 |
| pecora | 🐑 sheep | 1,800 |
| gatto | 🐱 cat | 1,700 |
| scoiattolo | 🐿️ squirrel | 1,500 |
| elefante | 🐘 elephant | 1,400 |

> ⚠️ The dataset folders are named in **Italian**. The app translates these to English automatically at prediction time.

### How to Download

**Option 1 — Browser:**
1. Go to https://www.kaggle.com/datasets/alessiocorrado99/animals10
2. Sign in with a free Kaggle account
3. Click **Download**
4. Extract and point your notebook to the `raw-img` folder

**Option 2 — Kaggle CLI:**
```bash
pip install kaggle
kaggle datasets download -d alessiocorrado99/animals10
unzip animals10.zip
```

---

## 📁 Project Structure

```
animal-classifier/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── mobilenet_animals_1_2.pth     # Trained MobileNetV2 weights
├── shufflenet_animals_1_2.pth    # Trained ShuffleNetV2 weights
│
├── Training_eval_phase2.ipynb    # Training & evaluation notebook
├── Hyperparameter_tuning_phase2.ipynb  # Grid search notebook
│
└── README.md                     # This file
```

> 💡 All four files (`app.py`, `requirements.txt`, and both `.pth` files) must be in the **same directory** for the app to run correctly.

---

## 🔧 Installation Guide

### Prerequisites

Make sure you have the following installed:
- Python 3.10 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify model files are present

Make sure both `.pth` files are in the project root:
```
mobilenet_animals_1_2.pth
shufflenet_animals_1_2.pth
```

> If you don't have them, they should be downloaded from the repository directly (they are committed alongside the code since they are under 100 MB each).

---

## ▶️ How to Run Locally

```bash
streamlit run app.py
```

The app will open automatically in your browser at:
```
http://localhost:8501
```

**App flow:**
1. Select a model from the sidebar (MobileNet or ShuffleNet)
2. Upload any animal photo using the file uploader
3. Click **🔍 Predict**
4. View the predicted animal, confidence score, and top-3 results

---

## ☁️ Deploy on Streamlit Cloud

Streamlit Community Cloud lets you host your app **for free** with a public URL.

### Step 1 — Push everything to GitHub

Make sure your repo contains:
```
app.py
requirements.txt
mobilenet_animals_1_2.pth
shufflenet_animals_1_2.pth
```

### Step 2 — Sign up at Streamlit Cloud

Go to 👉 https://share.streamlit.io and sign in with your GitHub account.

### Step 3 — Deploy

1. Click **New app**
2. Select your repository and branch
3. Set **Main file path** to `app.py`
4. Click **Deploy**

Your app will be live in 2–3 minutes at a URL like:
```
https://your-app-name.streamlit.app
```

> 💡 **Tip:** If the app is sleeping (free tier), it wakes up automatically when someone visits the URL — just takes ~30 seconds the first time.

---

---

## 🔮 Future Improvements

- [ ] Add **Grad-CAM** visualization to highlight which part of the image the model focuses on
- [ ] Support **batch prediction** — upload multiple images at once
- [ ] Add a **confidence threshold warning** — alert the user when confidence is below 50%
- [ ] Train on a **larger, balanced dataset** to fix class imbalance (spider vs elephant)
- [ ] Experiment with **EfficientNet** or **ConvNeXt** for better accuracy
- [ ] Add a **model comparison mode** — run both models side by side on the same image
- [ ] Export predictions as a **downloadable CSV** report
- [ ] Add **dark mode** support to the UI

---

## 🙏 Acknowledgements

- Dataset by [Alessio Corrado](https://www.kaggle.com/alessiocorrado99) on Kaggle
- Pretrained model weights from [PyTorch TorchVision](https://pytorch.org/vision/stable/models.html)
- UI built with [Streamlit](https://streamlit.io)
- Fonts: [Syne](https://fonts.google.com/specimen/Syne) + [DM Sans](https://fonts.google.com/specimen/DM+Sans) via Google Fonts

---

<p align="center">Built with 🧠 and ☕ — feel free to fork, star, or reach out!</p>
