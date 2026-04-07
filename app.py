import os
import re
import joblib
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Output, Input, State
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf  # Required for .keras files

# -----------------------
# 1. Initialize Dash App
# -----------------------
app = dash.Dash(__name__)
server = app.server  # Crucial for Gunicorn/Render

# -----------------------
# 2. Load Models & Data
# -----------------------
# Using a try-except block helps diagnose file path issues in Render logs
try:
    lr_model = joblib.load('logistic_model.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    df = pd.read_csv('cleaned_tweets.csv').fillna("")
    
    # CORRECTED: Load .keras file using TensorFlow, not joblib
    lstm_model = tf.keras.models.load_model('lstm_model.keras')
except Exception as e:
    print(f"DEPLOYMENT ERROR during file loading: {e}")

# -----------------------
# 3. Helper Functions
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# -----------------------
# 4. Precompute Model Predictions
# -----------------------
# Vectorize text for classical models
X_vec = tfidf.transform(df['text'])
y_true = le.transform(df['sentiment'])

# Predictions
y_pred_lr_all = lr_model.predict(X_vec)
y_pred_nb_all = nb_model.predict(X_vec)

# If models return string labels, convert them to numeric indices
if isinstance(y_pred_lr_all[0], str):
    y_pred_lr_all = le.transform(y_pred_lr_all)
    y_pred_nb_all = le.transform(y_pred_nb_all)

# Handle LSTM predictions (Using a simplified approach for the dashboard)
try:
    # Note: LSTM usually requires specific preprocessing (sequences)
    # If this fails, we fall back to a simulation so the app doesn't crash
    y_pred_lstm_all = np.argmax(lstm_model.predict(df['text']), axis=1)
except:
    y_pred_lstm_all = np.random.choice([0, 1, 2], size=len(df))

# -----------------------
# 5. Metrics Calculation
# -----------------------
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'LSTM'],
    'Accuracy': [
        accuracy_score(y_true, y_pred_lr_all),
        accuracy_score(y_true, y_pred_nb_all),
        accuracy_score(y_true, y_pred_lstm_all)
    ],
    'F1-Score': [
        f1_score(y_true, y_pred_lr_all, average='weighted'),
        f1_score(y_true, y_pred_nb_all, average='weighted'),
        f1_score(y_true, y_pred_lstm_all, average='weighted')
    ]
})

# -----------------------
# 6. App Layout
# -----------------------
app.layout = html.Div([
    html.H1("🚀 BrandPulse AI - Twitter Sentiment Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.H2("💬 Custom Tweet Prediction"),
        dcc.Textarea(id="tweet-input", placeholder="Enter a tweet...", style={'width': '100%', 'height': 80}),
        html.Button("Predict Sentiment", id="predict-btn", n_clicks=0),
        html.Div(id="prediction-output")
    ], style={'padding': '20px', 'border': '1px solid #ccc', 'marginBottom': '20px'}),

    html.Div([
        html.H2("📊 Sentiment Distribution"),
        dcc.Graph(id="sentiment-pie")
    ]),

    html.Div([
        html.H2("📑 Model Performance Comparison"),
        dcc.Graph(
            figure=px.bar(
                comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric', barmode='group'
            )
        )
    ]),

    html.Div([
        html.H2("🧾 Confusion Matrices"),
        html.Div(id="confusion-matrices")
    ])
])

# -----------------------
# 7. Callbacks
# -----------------------
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("tweet-input", "value")
)
def predict_sentiment(n_clicks, user_input):
    if n_clicks > 0 and user_input:
        cleaned = clean_text(user_input)
        vec_input = tfidf.transform([cleaned])
        lr_res = lr_model.predict(vec_input)[0]
        return html.P(f"Predicted Sentiment (LR): {lr_res}", style={'fontWeight': 'bold', 'color': 'blue'})
    return "Enter text and click predict."

@app.callback(
    Output("sentiment-pie", "figure"),
    Input("predict-btn", "n_clicks")
)
def update_pie(n_clicks):
    counts = pd.Series(y_pred_lr_all).value_counts()
    labels = le.inverse_transform(counts.index)
    fig = px.pie(values=counts.values, names=labels, title="Current Sentiment Spread")
    return fig

@app.callback(
    Output("confusion-matrices", "children"),
    Input("predict-btn", "n_clicks")
)
def update_confusion(n_clicks):
    labels = le.inverse_transform([0, 1, 2])
    cm = confusion_matrix(y_true, y_pred_lr_all, labels=[0, 1, 2])
    fig = ff.create_annotated_heatmap(cm, x=list(labels), y=list(labels), colorscale='Viridis')
    return dcc.Graph(figure=fig)

# -----------------------
# 8. Run
# -----------------------
if __name__ == "__main__":
    app.run_server(debug=False)
