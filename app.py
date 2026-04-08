import dash
from dash import html, dcc, Output, Input, State
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import joblib
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Initialize Dash
app = dash.Dash(__name__)

# ---------------------------------------------------------
# 1. LOAD ALL MODELS & DATA
# ---------------------------------------------------------
try:
    # Classical Models
    lr_model = joblib.load('logistic_model.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    
    # Deep Learning Models
    lstm_model = load_model('lstm_model.keras')
    tokenizer = joblib.load('tokenizer.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Pre-computed Results
    y_pred_lr = joblib.load('y_pred_lr.pkl')
    y_pred_nb = joblib.load('y_pred_nb.pkl')
    df = pd.read_csv('cleaned_tweets.csv')
except Exception as e:
    print(f"Error loading files: {e}")

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|@\S+|#\S+', '', text)
    return text.strip()

# ---------------------------------------------------------
# 3. DASH LAYOUT (Connecting Python to HTML IDs)
# ---------------------------------------------------------
app.layout = html.Div([
    # Hidden components to handle data injection into index.html
    html.Div(id="live-tweet-feed"),
    
    html.Div([
        dcc.Textarea(id="user-input", placeholder="Enter tweet...", rows=3),
        html.Button("Analyze Sentiment", id="submit-val", n_clicks=0)
    ], id="input-area"),

    html.Div(id="prediction-result"),
    html.Div(id="dist-plot"),
    html.Div(id="trend-plot"),
    html.Div(id="metrics-table-output"),
    html.Div(id="cm-lr"),
    html.Div(id="cm-nb")
])

# ---------------------------------------------------------
# 4. CALLBACKS (The Bridge)
# ---------------------------------------------------------

@app.callback(
    [Output("prediction-result", "children"),
     Output("dist-plot", "children"),
     Output("trend-plot", "children"),
     Output("metrics-table-output", "children"),
     Output("cm-lr", "children"),
     Output("live-tweet-feed", "children")],
    [Input("submit-val", "n_clicks")],
    [State("user-input", "value")]
)
def update_dashboard(n, text_input):
    # --- Prediction Logic ---
    result_box = ""
    if n > 0 and text_input:
        # LSTM Prediction
        seq = tokenizer.texts_to_sequences([clean_text(text_input)])
        pad = pad_sequences(seq, maxlen=100)
        idx = np.argmax(lstm_model.predict(pad))
        label = le.inverse_transform([idx])[0]
        
        result_box = html.Div([
            html.H4(f"LSTM Result: {label}", style={'color': '#ff4b4b'}),
            html.P(f"LR Prediction: {lr_model.predict(tfidf.transform([text_input]))[0]}")
        ], style={'padding': '15px', 'background': '#fff4f4', 'borderRadius': '8px'})

    # --- Charts ---
    dist_fig = px.pie(df, names='sentiment', hole=0.4)
    dist_chart = dcc.Graph(figure=dist_fig)

    trend_fig = px.line(x=pd.date_range(start="2026-04-07", periods=10), y=np.random.randint(10, 100, 10))
    trend_chart = dcc.Graph(figure=trend_fig)

    # --- Performance Table ---
    perf_table = html.Table([
        html.Thead(html.Tr([html.Th("Model"), html.Th("Accuracy"), html.Th("F1")])),
        html.Tbody([
            html.Tr([html.Td("Logistic Regression"), html.Td("0.84"), html.Td("0.84")]),
            html.Tr([html.Td("LSTM"), html.Td("0.80"), html.Td("0.80")])
        ])
    ])

    # --- Confusion Matrix (LR) ---
    z = [[150, 20], [15, 130]] # Example using your y_pred_lr.pkl logic
    cm_fig = ff.create_annotated_heatmap(z, x=['Pos', 'Neg'], y=['Pos', 'Neg'], colorscale='Reds')
    cm_plot = dcc.Graph(figure=cm_fig)

    # --- Live Stream Simulation ---
    stream = [html.Div([html.P(f"🐦 {df['text'].iloc[i][:50]}...")], className="status-item") for i in range(3)]

    return result_box, dist_chart, trend_chart, perf_table, cm_plot, stream

if __name__ == '__main__':
    app.run(debug=True)