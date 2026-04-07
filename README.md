# 🐦 Twitter Sentiment Analysis

A machine learning project that classifies tweets as **Positive** or **Negative** using Natural Language Processing (NLP) and Logistic Regression, trained on the Sentiment140 dataset (1.6 million tweets).

---

## 📌 Project Overview

This project builds a tweet sentiment classifier from scratch — covering the full ML pipeline from raw text preprocessing to model training, evaluation, and deployment-ready model saving.

| | |
|---|---|
| **Dataset** | Sentiment140 — 1.6M labeled tweets |
| **Algorithm** | Logistic Regression |
| **Text Vectorization** | TF-IDF |
| **Language** | Python |
| **Environment** | Google Colab / Jupyter Notebook |

---

## 🗂️ Project Structure

```
📦 twitter-sentiment-analysis
 ┣ 📓 TWITTER_SENTIMENT_ANALYSIS.ipynb   # Main notebook
 ┣ 🧠 trained_model.sav                  # Saved trained model (Pickle)
 ┣ 📊 training_1600000_processed_noemoticon.csv  # Dataset
 ┗ 📄 README.md
```

---

## 🔄 Pipeline

```
Raw Tweets
    ↓
Text Preprocessing (regex, lowercasing, stopword removal, stemming)
    ↓
TF-IDF Vectorization
    ↓
Train / Test Split (80% / 20%)
    ↓
Logistic Regression Training
    ↓
Evaluation & Model Saving
```

---

## 🧹 Text Preprocessing

Each tweet goes through the following steps:

1. **Remove non-alphabetic characters** using regex
2. **Lowercase** all words
3. **Tokenize** into individual words
4. **Remove stopwords** (NLTK English stopwords)
5. **Apply Porter Stemming** to reduce words to root form

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn nltk
```

```python
import nltk
nltk.download('stopwords')
```

### Run the Notebook

1. Clone the repository
2. Open `TWITTER_SENTIMENT_ANALYSIS.ipynb` in Jupyter or Google Colab
3. Update the dataset path in the CSV loading cell
4. Run all cells

### Use the Saved Model

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Predict
prediction = loaded_model.predict(x_new)

if prediction[0] == 0:
    print('Negative Tweet 😠')
else:
    print('Positive Tweet 😊')
```

---

## 📊 Dataset

**Sentiment140** — sourced from Stanford's large-scale Twitter sentiment dataset.

| Label | Meaning | Count |
|-------|---------|-------|
| 0 | Negative Tweet | 800,000 |
| 1 | Positive Tweet | 800,000 |

> Original labels were `0` (negative) and `4` (positive). The `4` label was remapped to `1` for binary classification.

---

## 📈 Model Performance

| Split | Accuracy |
|-------|----------|
| Training | ~79% |
| Testing | ~77% |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightblue?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---
