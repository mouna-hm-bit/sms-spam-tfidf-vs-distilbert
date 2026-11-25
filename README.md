# sms-spam-tfidf-vs-distilbert

A comparative NLP study benchmarking **TF-IDF + Logistic Regression** against a fine-tuned **DistilBERT** (`distilbert-base-uncased`) transformer for SMS spam classification, including class-imbalance handling (SMOTE, class-weighted loss, and augmented text oversampling) and overfitting diagnostics.

## Overview

The notebook [`spam-sms.ipynb`](spam-sms.ipynb) evaluates two fundamentally different text classification paradigms on the **UCI SMS Spam Collection** dataset (`5,572` messages, heavily imbalanced toward `ham`):

1. **Classical Sparse Representation (CPU):** `TfidfVectorizer` (`stop_words="english"`, `max_features=5000`) paired with `LogisticRegression(max_iter=1000)`.
2. **Pretrained Transformer (Multi-GPU):** `DistilBertForSequenceClassification` (`distilbert-base-uncased`, `MAX_LEN=128`, `AdamW` with `lr=2e-5`, linear warmup scheduler), parallelized across **2× GPUs** with `torch.nn.DataParallel`.

Across three progressive experiments in the notebook, the models are compared on speed, accuracy, minority-class (`Spam`) precision/recall/F1, and generalization gap.

---

## Experiments & Results

### 1. Baseline (80/20 Stratified Split, No Resampling)
*Test set: 1,115 messages (966 Ham, 149 Spam)*

| Model | Hardware | Time | Test Accuracy | Spam Precision | Spam Recall | Spam F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | CPU | `0.11s` | 97.04% | 1.00 | 0.78 | 0.88 |
| **DistilBERT** (3 epochs) | 2× GPU | `83.06s` | **99.28%** | 0.99 | **0.96** | **0.97** |

> Without class balancing, TF-IDF + Logistic Regression achieves 1.00 spam precision but misses 22% of spam messages (`0.78` recall). DistilBERT raises spam recall to `0.96`.

---

### 2. Imbalance Handling + Overfitting Check (70/15/15 Split)
*Splits: 3,900 Train (`3,377 Ham` / `523 Spam`) · 836 Validation · 836 Test (`724 Ham` / `112 Spam`)*

- **Model 1 (`TF-IDF + SMOTE + Logistic Regression`):** Applies `SMOTE` strictly to the training vectors.
- **Model 2 (`DistilBERT + Weighted CrossEntropyLoss`):** Uses `compute_class_weight("balanced")` in `nn.CrossEntropyLoss` over 4 epochs.

| Model | Time | Test Accuracy | Spam Precision | Spam Recall | Spam F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + SMOTE + LogReg** | `0.14s` | 98.33% | 0.95 | 0.93 | 0.94 |
| **DistilBERT + Weighted Loss** (4 epochs) | `99.98s` | **99.40%** | **0.99** | **0.96** | **0.98** |

*Overfitting check (Model 1):* Train Acc `0.9913` vs. Val Acc `0.9785` (gap `0.0128`).

---

### 3. Augmented Text Oversampling + Learning Curves (70/15/15 Split)
- **Model 1 (`TF-IDF + SMOTE + LogReg`):** Resampled training set of `6,754` rows (`3,377 Ham` / `3,377 Spam`).
- **Model 2 (`DistilBERT + Augmented Oversampling`):** Balances the training text with `RandomOverSampler` (`6,754` rows) and applies light text augmentation (10% random word dropout + 50% chance of adjacent word swap) to oversampled `Spam` copies to reduce exact-duplicate memorization.

| Model | Time | Test Accuracy | Spam Precision | Spam Recall | Spam F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + SMOTE + LogReg** | `0.15s` | 98.33% | 0.95 | 0.93 | 0.94 |
| **DistilBERT + Augmented Oversampling** (4 epochs) | `171.64s` | **99.52%** | **1.00** | **0.96** | **0.98** |

*DistilBERT Epoch Progression:*
- Epoch 1: Train Loss `0.1418`, Val Loss `0.1035`, Val Acc `0.9725`
- Epoch 2: Train Loss `0.0211`, Val Loss `0.0612`, Val Acc `0.9868`
- Epoch 3: Train Loss `0.0065`, Val Loss `0.0561`, Val Acc `0.9928`
- Epoch 4: Train Loss `0.0037`, Val Loss `0.0588`, Val Acc `0.9904`

---

## Key Takeaways

- **Speed vs. Accuracy Trade-off:** TF-IDF + SMOTE + Logistic Regression trains and evaluates in **~0.15 seconds** on CPU and reaches **98.33%** test accuracy (`0.94` Spam F1).
- **Best Detection Performance:** Fine-tuned DistilBERT with augmented oversampling reaches **99.52%** test accuracy with **1.00** Spam precision and **0.96** Spam recall, at the cost of GPU training time (~172s on 2× T4 GPUs).
- **Handling Imbalance Matters:** Adding SMOTE lifted TF-IDF + Logistic Regression's Spam recall from `0.78` to `0.93` (+15 percentage points).

---

## Project Structure

```text
.
├── spam-sms.ipynb   # Full training, evaluation, learning curves, and confusion matrices
└── README.md        # Project overview and experimental results
```

## Requirements & Usage

### Dependencies
- Python 3
- `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`
- `torch`, `transformers`, `tqdm`
- `matplotlib`, `seaborn`

### Dataset
The notebook was authored for Kaggle using the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset):
```python
path = "/kaggle/input/datasets/uciml/sms-spam-collection-dataset/spam.csv"
```
If running locally, download `spam.csv` and update the file path in [`spam-sms.ipynb`](spam-sms.ipynb) before executing the cells.
