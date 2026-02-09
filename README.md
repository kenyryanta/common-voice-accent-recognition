# 🎙️ Wav2Vec 2.0 Accent Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-Mozilla%20Common%20Voice-green.svg)](https://commonvoice.mozilla.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

## 📖 Overview
This repository implements a rigorous data preprocessing pipeline aimed at building an **Accent Classification System** using Deep Learning. The project focuses on distinguishing between three major English dialects: **United States (US)**, **United Kingdom (England)**, and **Australia (AU)**.

The pipeline is engineered to ingest raw audio from the **Mozilla Common Voice** dataset, handle severe class imbalances, and transform audio signals into normalized vector inputs compatible with the **Wav2Vec 2.0** architecture.

## 🏗️ Architecture & Methodology

The pipeline follows a strict ETL (Extract, Transform, Load) procedure designed for audio machine learning:

### 1. Data Ingestion (Extract)
- **Source:** Mozilla Common Voice Dataset (via Kaggle Hub).
- **Format:** MP3 audio files with associated metadata (CSV).
- **Target Accents:**
  - `us` (American English)
  - `england` (British English)
  - `australia` (Australian English)

### 2. Data Cleaning & Balancing (Transform Phase I)
- **Filtering:** The script isolates specific accents from the multilingual dataset.
- **Class Balancing:** To mitigate model bias (where the model preferentially predicts the majority class), we apply **Random Under-Sampling**.
  - *Strategy:* Limit each accent class to $N=2500$ distinct samples.
  - *Result:* A perfectly balanced dataset ensuring equal contribution to the loss gradient during training.

### 3. Signal Processing (Transform Phase II)
- **Resampling:** All audio is resampled to **16kHz** ($16,000$ samples/second).
  - *Why?* The Wav2Vec 2.0 model was pre-trained on 16kHz audio. Mismatched sampling rates result in severe performance degradation.
- **Truncation:** Audio clips are truncated to a maximum of **5.0 seconds**.
  - *Why?* Transformer models have a quadratic memory complexity relative to sequence length. Capping duration ensures the model fits within standard GPU VRAM (e.g., T4/P100).
- **Feature Extraction:** Raw waveforms are normalized to zero mean and unit variance using `Wav2Vec2FeatureExtractor`.

### 4. Dataset Splitting
- **Training Set:** 80% (Used for model weight updates).
- **Testing Set:** 20% (Used for final evaluation metrics).

## 🛠️ Prerequisites & Installation

### Environment
The code is optimized for **Linux/Cloud environments** (e.g., Google Colab, AWS SageMaker) with GPU acceleration enabled.

### Dependencies
Install the required Python libraries:

```bash
pip install --upgrade pip
pip install transformers datasets librosa torch torchaudio scikit-learn accelerate kagglehub
