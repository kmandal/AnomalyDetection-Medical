# Medical Anomaly Detection on MedMNIST using VAEs

An open-source, modular framework for unsupervised anomaly detection on histopathology tissue slides using Variational Autoencoders (VAEs). This repository implements a fully generalized data and model pipeline allowing seamless evaluation between Dense (Linear) and Spatial Convolutional architectures.

## 📌 Project Overview
Unsupervised anomaly detection in medical imaging leverages models trained **strictly on healthy tissue** to identify deviations (such as cancer epithelium/stroma) by observing reconstruction spikes. 

### Current Status: Phase 1 Complete 
* **Generic Data Pipeline:** Decouples structural memory layouts from model configurations.
* **Unified Scripts:** Single flag controls for training, evaluating, and running hyperparameter grid sweeps across multiple architectures.
* **Baseline Validation:** Evaluated on **PathMNIST** (colon mucosa tissue slides).

---

## 🛠️ System Architecture & Modularity

The framework is structurally decoupled so that preprocessing adjustments do not require code rewriting at the model tier:
1. **`prepare_data.py`**: Saves raw 3D images `(N, 28, 28, 3)` to preserve spatial integrity.
2. **`train.py` / `evaluate.py`**: Handles dynamic runtime transformations:
   * `linear`: Flattened memory block configuration `(N, 2352)` for Dense VAEs.
   * `conv`: Channel-First memory permutation `(N, 3, 28, 28)` required for optimized GPU-coalesced 2D Convolutions.

---

## 🔬 Phase 1 Baseline Results & The VAE Paradox

Extensive hyperparameter optimization sweeps (varying `latent_dim` from 2 to 16 and `beta` from 0.5 to 20.0) reveal a stable **Separation Indicator** of **1.04x** for both linear and convolutional VAEs.

### The VAE Over-generalization Paradox
While the pipeline functions without errors, standard VAEs trained with pixel-wise Binary Cross Entropy (BCE) suffer from *over-generalization*. Because healthy and cancerous histopathology tissue share highly similar low-level pixel distributions and color profiles, a powerful unconstrained decoder effortlessly reconstructs anomalous structures. Standard pixel-level metrics fail to capture chaotic structural deformations without structural or perceptual constraints.

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com
cd medical-vae-anomaly
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Generate Data Splits
```bash
python src/prepare_data.py
```

### 3. Execution via Unified Flags
Train and evaluate a **Dense Linear VAE**:
```bash
python src/train.py --model linear
python src/evaluate.py --model linear
```

Train and evaluate a **Spatial Convolutional VAE**:
```bash
python src/train.py --model conv
python src/evaluate.py --model conv
```

### 4. Hyperparameter Optimization Sweep
Run an automated grid search across Latent Dimensions and Beta values:
```bash
python src/tune_vae.py --model conv
```

---

## 🗺️ Roadmap: Phase 2 and Beyond

To overcome pixel-level over-generalization and reliably separate structural tissue anomalies, Phase 2 will introduce advanced structural loss functions and alternative out-of-distribution paradigms:

- [ ] **Structural & Perceptual Objectives:** Integrating **Structural Similarity Index (SSIM)** and VGG/ResNet deep feature perceptual losses to heavily penalize cell architectural alterations.
- [ ] **Normalizing Flows:** Transitioning to exact likelihood estimators (e.g., RealNVP, Glow) to model complex density boundaries explicitly.
- [ ] **Quantile Regression Neural Networks:** Exploring conditional quantile bounds to isolate pixel and structural outliers rigorously.

---

## 🤝 Contributing
Contributions, issue reports, and structural loss recommendations are highly welcome! Please open an issue or submit a pull request if you want to collaborate on Phase 2 developments.
write something
