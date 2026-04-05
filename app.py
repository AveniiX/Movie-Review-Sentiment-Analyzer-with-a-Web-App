import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Loading model and vectorizer
@st.cache_resource
def load_model():
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().strip()
    words = [w for w in text.split()
             if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)

def predict_sentiment(text, model, vectorizer):
    cleaned  = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred     = model.predict(features)[0]
    prob     = model.predict_proba(features)[0]
    return pred, prob

# UI
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Paste any movie review and the model will predict if it's **positive** or **negative**.")

model, vectorizer = load_model()

review = st.text_area(
    "Enter a movie review:",
    height=180,
    placeholder="This movie was absolutely fantastic! The acting..."
)

if st.button("Analyze Sentiment", type="primary"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        pred, prob = predict_sentiment(review, model, vectorizer)

        if pred == 1:
            st.success(f"Positive Review — {prob[1]*100:.1f}% confidence")
        else:
            st.error(f"Negative Review — {prob[0]*100:.1f}% confidence")

        col1, col2 = st.columns(2)
        col1.metric("Positive", f"{prob[1]*100:.1f}%")
        col2.metric("Negative", f"{prob[0]*100:.1f}%")

st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit | IMDB 50K Dataset")