# Movie Review Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Accuracy](https://img.shields.io/badge/Accuracy-89%25-green)

A machine learning model that classifies movie reviews as
positive or negative using NLP techniques, deployed as a
live interactive web application.

**[Live Demo](https://movie-review-sentiment-analyzer-with-a-web-app-fimymdvsfyagxv9.streamlit.app/)**

---

## Overview

This project builds an end-to-end NLP pipeline:
raw movie review text → cleaned features → trained classifier
→ deployed web app.

**Dataset**: IMDB 50K Movie Reviews (Kaggle)
- 50,000 reviews, perfectly balanced (25k pos / 25k neg)
- Binary sentiment classification task

**Model**: Logistic Regression with TF-IDF features
- Accuracy: ~89% on test set
- Trained on 40,000 reviews, tested on 10,000

---

## Results

| Model               | Accuracy | Precision | Recall |
|---------------------|----------|-----------|--------|
| Logistic Regression | 89.2%    | 89.4%     | 89.1%  |
| Naive Bayes         | 86.8%    | 87.1%     | 86.5%  |

Logistic Regression outperformed Naive Bayes by ~2.4%.

---

## Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data manipulation
- **NLTK** — text preprocessing, stopword removal
- **Scikit-learn** — TF-IDF vectorization, model training
- **Matplotlib / Seaborn** — EDA and visualization
- **Streamlit** — web app deployment
- **Pickle** — model serialization

---

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analyzer
cd sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
sentiment-analyzer/
├── notebook.ipynb     # Full ML pipeline with EDA
├── app.py             # Streamlit web application
├── model/
│   ├── model.pkl      # Trained Logistic Regression
│   └── vectorizer.pkl # TF-IDF vectorizer
└── requirements.txt
```

---

## Key Learnings

- TF-IDF with bigrams (ngram_range=(1,2)) improved accuracy
  by ~1.5% over unigrams alone
- Removing HTML tags was critical — IMDB reviews contain <br> tags
- Logistic Regression is fast, interpretable, and performs
  surprisingly well on text classification tasks
- Model serialization (pickle) is essential for deployment

---

## Author

**Muhammad Hamam** — Data Scientist
[LinkedIn](https://www.linkedin.com/in/muhammad-hamam-yousif-b90455374/) | [GitHub](https://github.com/AveniiX)
