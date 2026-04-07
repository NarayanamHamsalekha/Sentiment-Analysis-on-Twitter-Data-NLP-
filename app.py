# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------
# Load Models
# -----------------------
st.sidebar.title("Model Loading Status")

try:
    lr_model = joblib.load('logistic_model.pkl')
    st.sidebar.success("✅ Logistic Regression Loaded")
except:
    lr_model = None
    st.sidebar.warning("⚠️ Logistic Regression Not Found")

try:
    nb_model = joblib.load('naive_bayes_model.pkl')
    st.sidebar.success("✅ Naive Bayes Loaded")
except:
    nb_model = None
    st.sidebar.warning("⚠️ Naive Bayes Not Found")

try:
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    st.sidebar.success("✅ TF-IDF Vectorizer Loaded")
except:
    tfidf = None
    st.sidebar.warning("⚠️ TF-IDF Vectorizer Not Found")

try:
    le = joblib.load('label_encoder.pkl')
    st.sidebar.success("✅ Label Encoder Loaded")
except:
    le = None
    st.sidebar.warning("⚠️ Label Encoder Not Found")

try:
    df = pd.read_csv('cleaned_tweets.csv')
    df['text'] = df['text'].fillna("")
    st.sidebar.success("✅ Dataset Loaded")
except:
    df = pd.DataFrame(columns=['text', 'sentiment'])
    st.sidebar.warning("⚠️ Dataset Not Found")

# -----------------------
# UI
# -----------------------
st.title("🚀 BrandPulse AI - Twitter Sentiment Dashboard")
st.markdown("### Analyze Twitter Sentiments in Real-Time")
st.write("---")

# -----------------------
# Text Cleaning
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# -----------------------
# Prediction
# -----------------------
st.header("💬 Custom Tweet Sentiment Prediction")
user_input = st.text_area("Enter a Tweet:")

if st.button("Predict Sentiment"):
    if user_input and tfidf and lr_model and nb_model:
        cleaned_input = clean_text(user_input)
        vec_input = tfidf.transform([cleaned_input])

        lr_pred = lr_model.predict(vec_input)[0]
        nb_pred = nb_model.predict(vec_input)[0]

        st.success(f"🎯 Logistic Regression: {lr_pred}")
        st.success(f"🎯 Naive Bayes: {nb_pred}")
    else:
        st.warning("Please enter tweet or models not loaded")

st.write("---")

# -----------------------
# Live Tweets
# -----------------------
st.header("📢 Live Tweet Stream (Simulation)")
if not df.empty:
    for i, tweet in enumerate(df.sample(5)['text'], start=1):
        st.write(f"{i}. {tweet}")

st.write("---")

# -----------------------
# Sentiment Distribution
# -----------------------
st.header("📊 Sentiment Distribution")
if not df.empty and le:
    counts = df['sentiment'].value_counts()
    st.bar_chart(counts)
else:
    st.info("No data available")

st.write("---")

# -----------------------
# Trend Line
# -----------------------
st.header("📈 Sentiment Trend (24 Hours)")
timestamps = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='H')
trend_data = np.random.randint(0, 50, size=(24, 3))
trend_df = pd.DataFrame(trend_data, columns=['negative','neutral','positive'], index=timestamps)
st.line_chart(trend_df)

st.write("---")

# -----------------------
# Performance Metrics
# -----------------------
st.header("📑 Performance Comparison")

if not df.empty and le and tfidf and lr_model and nb_model:
    y_true = le.transform(df['sentiment'])

    y_pred_lr = le.transform(lr_model.predict(tfidf.transform(df['text'])))
    y_pred_nb = le.transform(nb_model.predict(tfidf.transform(df['text'])))

    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Naive Bayes'],
        'Accuracy': [
            accuracy_score(y_true, y_pred_lr),
            accuracy_score(y_true, y_pred_nb)
        ],
        'Precision': [
            precision_score(y_true, y_pred_lr, average='weighted'),
            precision_score(y_true, y_pred_nb, average='weighted')
        ],
        'Recall': [
            recall_score(y_true, y_pred_lr, average='weighted'),
            recall_score(y_true, y_pred_nb, average='weighted')
        ],
        'F1-Score': [
            f1_score(y_true, y_pred_lr, average='weighted'),
            f1_score(y_true, y_pred_nb, average='weighted')
        ]
    })

    st.dataframe(comparison)

st.write("---")

# -----------------------
# Confusion Matrix
# -----------------------
st.header("📊 Confusion Matrices")

labels = ['negative', 'neutral', 'positive']

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels).reset_index().melt(id_vars='index')
    cm_df.columns = ['Actual', 'Predicted', 'Count']

    chart = alt.Chart(cm_df).mark_rect().encode(
        x='Predicted:N',
        y='Actual:N',
        color='Count:Q',
        tooltip=['Actual', 'Predicted', 'Count']
    ).properties(title=title)

    st.altair_chart(chart)

if not df.empty and le:
    y_true_labels = df['sentiment']
    y_pred_lr_labels = lr_model.predict(tfidf.transform(df['text']))
    y_pred_nb_labels = nb_model.predict(tfidf.transform(df['text']))

    plot_cm(y_true_labels, y_pred_lr_labels, "Logistic Regression")
    plot_cm(y_true_labels, y_pred_nb_labels, "Naive Bayes")
