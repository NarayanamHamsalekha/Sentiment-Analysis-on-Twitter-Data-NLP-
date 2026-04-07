import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib

# Load models
lr_model = joblib.load('logistic_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Load dataset
df = pd.read_csv('cleaned_tweets.csv')

# Create app
app = dash.Dash(__name__)
server = app.server  # IMPORTANT for Render

# Layout
app.layout = html.Div([
    html.H1("🚀 BrandPulse AI - Sentiment Dashboard"),

    dcc.Textarea(
        id='tweet-input',
        placeholder='Enter a tweet...',
        style={'width': '100%', 'height': 100}
    ),

    html.Button('Predict', id='predict-btn'),

    html.Div(id='output'),

    html.H3("📊 Sentiment Distribution"),

    dcc.Graph(
        id='sentiment-graph',
        figure={
            'data': [{
                'labels': df['sentiment'].value_counts().index,
                'values': df['sentiment'].value_counts().values,
                'type': 'pie'
            }]
        }
    )
])

# Callback
@app.callback(
    Output('output', 'children'),
    Input('predict-btn', 'n_clicks'),
    Input('tweet-input', 'value')
)
def predict(n_clicks, text):
    if n_clicks and text:
        vec = tfidf.transform([text])
        lr_pred = lr_model.predict(vec)[0]
        nb_pred = nb_model.predict(vec)[0]

        return html.Div([
            html.P(f"Logistic Regression: {lr_pred}"),
            html.P(f"Naive Bayes: {nb_pred}")
        ])
    return ""

# Run app
if __name__ == '__main__':
    app.run(debug=True)