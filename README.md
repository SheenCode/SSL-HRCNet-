# SSL-HRCNet: A Self-Supervised Learning Framework with Hierarchical Residual Cross Fusion Network for Sleep Apnea Detection

## 📌 Overview

Sleep apnea is a prevalent yet underdiagnosed respiratory disorder associated with significant health risks, including cardiovascular diseases and cognitive impairments. This repository provides the implementation of **SSL-HRCNet**, a novel self-supervised learning framework designed for sleep apnea detection using single-lead ECG signals.

The proposed framework addresses the challenge of limited labeled data by leveraging physiologically meaningful transformations derived from ECG signals, enabling robust and scalable representation learning.

---

## 🧠 Key Contributions

- **Self-Supervised Learning Framework**
  - A two-stage pipeline: contrastive pre-training + supervised fine-tuning
  - Reduces dependence on large-scale annotated datasets

- **Physiologically Meaningful Contrastive Learning**
  - Uses **R-R intervals (RRIs)** and **R-peak amplitudes (R-peaks)** as two complementary views
  - Captures cardiopulmonary dynamics rather than relying on generic augmentations

- **Hierarchical Learnable Residual Encoder**
  - Depthwise–pointwise convolution blocks
  - Multi-scale temporal modeling (kernel sizes: 3, 5, 7)
  - Learnable residual connections for adaptive feature flow

- **Attention-based Cross Fusion Module**
  - Bidirectional interaction between RRIs and R-peaks
  - Enhances feature representation for apnea detection

- **Strong Generalization Ability**
  - Effective with limited labeled data (e.g., 10%)
  - Robust transfer performance across datasets (Apnea-ECG → UCDDB)

---

## 🏗️ Model Architecture

The framework consists of two main components:

### 1. Self-Supervised Pre-training
- Input: unlabeled ECG signals
- Generate two views:
  - RRIs
  - R-peaks
- Contrastive learning to align physiological representations

### 2. Supervised Fine-tuning
- Hierarchical residual encoder extracts multi-scale features
- Cross-attention fusion integrates RRIs and R-peaks
- Final classifier predicts apnea events

---

## 📊 Datasets

- **Apnea-ECG Dataset** (PhysioNet)
- **UCDDB Dataset**

⚠️ Note:  
Due to data privacy policies, datasets are not included in this repository.  
Please download them from official sources:

- https://physionet.org/

