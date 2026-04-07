# app_dash_full.py
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
import re

# -----------------------
# Load Models & Data
# -----------------------
lr_model = joblib.load('logistic_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
lstm_model = load_model('lstm_model.keras')
tokenizer = joblib.load('tokenizer.pkl')
le = joblib.load('label_encoder.pkl')
df = pd.read_csv('cleaned_tweets.csv').fillna("")

# -----------------------
# Text Cleaning
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# Precompute Model Predictions
# -----------------------
# TF-IDF vectors
X_vec = tfidf.transform(df['text'])
y_true = le.transform(df['sentiment'])

y_pred_lr_all = le.transform(lr_model.predict(X_vec))
y_pred_nb_all = le.transform(nb_model.predict(X_vec))

# LSTM predictions
X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=100)
y_pred_lstm_all = np.argmax(lstm_model.predict(X_pad), axis=1)

# Performance Metrics Table
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'LSTM'],
    'Accuracy': [
        accuracy_score(y_true, y_pred_lr_all),
        accuracy_score(y_true, y_pred_nb_all),
        accuracy_score(y_true, y_pred_lstm_all)
    ],
    'Precision': [
        precision_score(y_true, y_pred_lr_all, average='weighted'),
        precision_score(y_true, y_pred_nb_all, average='weighted'),
        precision_score(y_true, y_pred_lstm_all, average='weighted')
    ],
    'Recall': [
        recall_score(y_true, y_pred_lr_all, average='weighted'),
        recall_score(y_true, y_pred_nb_all, average='weighted'),
        recall_score(y_true, y_pred_lstm_all, average='weighted')
    ],
    'F1-Score': [
        f1_score(y_true, y_pred_lr_all, average='weighted'),
        f1_score(y_true, y_pred_nb_all, average='weighted'),
        f1_score(y_true, y_pred_lstm_all, average='weighted')
    ]
})

# -----------------------
# Dash App Setup
# -----------------------
app1 = dash.Dash(__name__)
server = app1.server

app1.layout = html.Div([
    html.H1("🚀 BrandPulse AI - Twitter Sentiment Dashboard"),
    
    # Custom Tweet Input
    html.Div([
        html.H2("💬 Custom Tweet Prediction"),
        dcc.Textarea(id="tweet-input", placeholder="Enter a tweet...", style={'width': '100%', 'height': 80}),
        html.Button("Predict Sentiment", id="predict-btn", n_clicks=0),
        html.Div(id="prediction-output")
    ], style={'margin-bottom': '50px'}),

    # Live Tweet Stream
    html.Div([
        html.H2("📢 Live Tweet Stream (Simulation)"),
        html.Ul(id="tweet-stream", children=[])
    ], style={'margin-bottom': '50px'}),

    # Sentiment Distribution Pie Chart
    html.Div([
        html.H2("📊 Sentiment Distribution"),
        dcc.Graph(id="sentiment-pie")
    ], style={'margin-bottom': '50px'}),

    # Trend Line
    html.Div([
        html.H2("📈 Sentiment Trend Over Last 24 Hours (Simulation)"),
        dcc.Graph(id="trend-line")
    ], style={'margin-bottom': '50px'}),

    # Performance Comparison Table
    html.Div([
        html.H2("📑 Model Performance Comparison"),
        dcc.Graph(
            figure=px.bar(
                comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric', barmode='group', title="Model Performance Metrics"
            )
        )
    ], style={'margin-bottom': '50px'}),

    # Confusion Matrices
    html.Div([
        html.H2("🧾 Confusion Matrices"),
        html.Div(id="confusion-matrices")
    ])
])

# -----------------------
# Callbacks
# -----------------------
# Custom Tweet Prediction
@app1.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("tweet-input", "value")
)
def predict_sentiment(n_clicks, user_input):
    if n_clicks > 0 and user_input:
        cleaned = clean_text(user_input)
        vec_input = tfidf.transform([cleaned])
        lr_pred = lr_model.predict(vec_input)[0]
        nb_pred = nb_model.predict(vec_input)[0]
        seq = tokenizer.texts_to_sequences([cleaned])
        pad_seq = pad_sequences(seq, maxlen=100)
        lstm_class = np.argmax(lstm_model.predict(pad_seq), axis=1)
        lstm_pred = le.inverse_transform(lstm_class)[0]
        return html.Div([
            html.P(f"✅ LSTM Prediction: {lstm_pred}"),
            html.P(f"Naive Bayes Prediction: {nb_pred}"),
            html.P(f"Logistic Regression Prediction: {lr_pred}")
        ])
    return ""

# Live Tweet Stream
@app1.callback(
    Output("tweet-stream", "children"),
    Input("predict-btn", "n_clicks")
)
def update_stream(n_clicks):
    sample = df.sample(5)['text'].tolist()
    return [html.Li(tweet) for tweet in sample]

# Sentiment Pie Chart
@app1.callback(
    Output("sentiment-pie", "figure"),
    Input("predict-btn", "n_clicks")
)
def update_pie(n_clicks):
    counts = pd.Series(y_pred_lstm_all).value_counts()
    labels = le.inverse_transform(counts.index)
    pie_df = pd.DataFrame({"Sentiment": labels, "Count": counts.values})
    fig = px.pie(pie_df, names="Sentiment", values="Count", title="Tweet Sentiment Distribution")
    return fig

# Trend Line
@app1.callback(
    Output("trend-line", "figure"),
    Input("predict-btn", "n_clicks")
)
def update_trend(n_clicks):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='H')
    sentiment_counts = np.random.randint(0, 50, size=(24, 3))
    trend_df = pd.DataFrame(sentiment_counts, columns=le.inverse_transform([0,1,2]), index=timestamps)
    fig = px.line(trend_df, x=trend_df.index, y=trend_df.columns, labels={"value": "Count", "index": "Time"})
    return fig

# Confusion Matrices
@app1.callback(
    Output("confusion-matrices", "children"),
    Input("predict-btn", "n_clicks")
)
def update_confusion(n_clicks):
    labels = le.inverse_transform([0,1,2])
    cm_list = []
    for model_name, y_pred in [("Logistic Regression", y_pred_lr_all),
                               ("Naive Bayes", y_pred_nb_all),
                               ("LSTM", y_pred_lstm_all)]:
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        fig = ff.create_annotated_heatmap(cm, x=labels, y=labels, colorscale='Viridis', showscale=True)
        fig.update_layout(title_text=f"{model_name} Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        cm_list.append(dcc.Graph(figure=fig))
    return cm_list

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app1.run_server(debug=True)
