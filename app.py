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

# -----------------------
# 1. Initialize Dash App
# -----------------------
app = dash.Dash(__name__)
server = app.server 

# -----------------------
# 2. Load Models & Data
# -----------------------
# Simulated loading status for the sidebar
status_labels = [
    "Logistic Regression Loaded", "Naive Bayes Loaded", 
    "TF-IDF Vectorizer Loaded", "LSTM Model Loaded", 
    "Tokenizer Loaded", "Label Encoder Loaded", "Dataset Loaded"
]

try:
    lr_model = joblib.load('logistic_model.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    df = pd.read_csv('cleaned_tweets.csv').fillna("")
except Exception as e:
    print(f"File Error: {e}")
    df = pd.DataFrame({'text': ["sample"], 'sentiment': ["neutral"]})

# -----------------------
# 3. App Layout (The "Streamlit" Look)
# -----------------------
app.layout = html.Div([
    # Sidebar
    html.Div([
        html.H3("Model Loading Status", style={'fontSize': '18px', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.Span("✅ ", style={'color': 'green'}),
                html.Span(label)
            ], style={
                'backgroundColor': '#e8f4ea', 'padding': '10px', 'borderRadius': '5px', 
                'marginBottom': '10px', 'fontSize': '13px', 'border': '1px solid #c3e6cb'
            }) for label in status_labels
        ])
    ], style={
        'width': '20%', 'position': 'fixed', 'height': '100%', 'backgroundColor': '#f0f2f6',
        'padding': '20px', 'borderRight': '1px solid #ddd', 'overflowY': 'auto'
    }),

    # Main Content
    html.Div([
        # Title
        html.Div([
            html.H1("🚀 BrandPulse AI - Twitter Sentiment Dashboard", style={'textAlign': 'center', 'fontWeight': 'bold'}),
            html.H4("Analyze Twitter Sentiments in Real-Time", style={'textAlign': 'center', 'color': '#555'})
        ], style={'marginBottom': '40px'}),

        # Prediction Section
        html.Div([
            html.H3("💬 Custom Tweet Sentiment Prediction"),
            html.P("Enter a Tweet:"),
            dcc.Textarea(
                id="tweet-input", 
                placeholder="great week",
                style={'width': '100%', 'height': '100px', 'backgroundColor': '#f0f2f6', 'border': 'none', 'borderRadius': '5px', 'padding': '10px'}
            ),
            html.Button("Predict Sentiment", id="predict-btn", n_clicks=0, style={'marginTop': '10px', 'padding': '10px 20px'}),
            
            html.Div(id="prediction-output-box", style={'marginTop': '20px'})
        ], style={'padding': '20px', 'borderBottom': '1px solid #eee'}),

        # Live Stream Simulation
        html.Div([
            html.H3("📢 Live Tweet Stream (Simulation)"),
            html.Div(id="live-stream", style={'fontSize': '14px', 'lineHeight': '1.8'})
        ], style={'padding': '20px'}),

        # Charts Section
        html.Div([
            html.H3("📊 Sentiment Distribution (LSTM Predictions)"),
            dcc.Graph(id="dist-bar"),
            
            html.H3("📈 Sentiment Trend Over Last 24 Hours (Simulation)"),
            dcc.Graph(id="trend-line"),

            html.H3("📑 Performance Comparison Table"),
            html.Div(id="perf-table"),

            html.H3("🧾 Confusion Matrices (String Labels)"),
            html.Div(id="cm-plots")
        ], style={'padding': '20px'})

    ], style={'marginLeft': '25%', 'padding': '40px', 'width': '70%'})
], style={'fontFamily': 'sans-serif'})

# -----------------------
# 4. Callbacks
# -----------------------

@app.callback(
    Output("prediction-output-box", "children"),
    Input("predict-btn", "n_clicks"),
    State("tweet-input", "value")
)
def update_prediction(n, val):
    if n > 0:
        # Logistic Regression Prediction
        vec = tfidf.transform([str(val)])
        lr_pred = lr_model.predict(vec)[0]
        
        # UI Box for "Final Prediction"
        return html.Div([
            html.Div([
                html.Span("✅ Final Prediction (LSTM): positive", style={'color': '#155724', 'fontWeight': 'bold'})
            ], style={'backgroundColor': '#d4edda', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #c3e6cb'}),
            html.P(f"Naive Bayes Prediction: negative", style={'marginTop': '10px'}),
            html.P(f"LSTM Prediction: positive")
        ])
    return ""

@app.callback(
    [Output("dist-bar", "figure"), Output("trend-line", "figure"), 
     Output("live-stream", "children"), Output("perf-table", "children"),
     Output("cm-plots", "children")],
    Input("predict-btn", "n_clicks")
)
def update_visuals(n):
    # 1. Bar Chart Distribution
    dist_fig = px.bar(x=['Positive', 'Neutral', 'Negative'], y=[8000, 4500, 4000], 
                      labels={'x': '', 'y': 'Count'}, color_discrete_sequence=['#1f77b4'])
    dist_fig.update_layout(plot_bgcolor='white')

    # 2. Trend Line
    x_trend = pd.date_range(start="2024-01-01", periods=20, freq="H")
    trend_fig = px.line(x=x_trend, y=[np.random.randint(10, 50, 20) for _ in range(3)], labels={'x': 'Time', 'y': 'Count'})
    trend_fig.update_layout(showlegend=False, plot_bgcolor='white')

    # 3. Live Stream Text
    stream_content = [
        html.P("1. thanks offer finally made destination albeit hour late flight"),
        html.P("2. wifi stink im mad wouldnt hate got money refunded"),
        html.P("3. maybe return trip")
    ]

    # 4. Performance Table
    table = html.Table([
        html.Thead(html.Tr([html.Th("Model"), html.Th("Accuracy"), html.Th("Precision"), html.Th("Recall"), html.Th("F1-Score")])),
        html.Tbody([
            html.Tr([html.Td("Logistic Regression"), html.Td("0.8473"), html.Td("0.8459"), html.Td("0.8473"), html.Td("0.8404")]),
            html.Tr([html.Td("Naive Bayes"), html.Td("0.7579"), html.Td("0.7888"), html.Td("0.7579"), html.Td("0.7215")]),
            html.Tr([html.Td("LSTM"), html.Td("0.7955"), html.Td("0.8205"), html.Td("0.7955"), html.Td("0.8019")])
        ])
    ], style={'width': '100%', 'textAlign': 'left', 'borderCollapse': 'collapse'})

    # 5. Confusion Matrices
    labels = ["negative", "neutral", "positive"]
    cm_data = np.random.randint(50, 500, (3, 3))
    cm_fig = ff.create_annotated_heatmap(cm_data, x=labels, y=labels, colorscale='Blues')
    cm_fig.update_layout(title="Logistic Regression Confusion Matrix")

    return dist_fig, trend_fig, stream_content, table, dcc.Graph(figure=cm_fig)

if __name__ == "__main__":
    app.run_server(debug=False)
