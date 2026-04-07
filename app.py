# app.py
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ----------------------
# Initialize Dash App
# ----------------------
app = dash.Dash(__name__)
server = app.server  # Needed for deployment

# ----------------------
# Simulate Tweet Stream
# ----------------------
# Example dataset for simulation
sentiment_classes = ["Positive", "Neutral", "Negative"]

# Create an initial DataFrame for past 24 hours
time_now = datetime.now()
timestamps = [time_now - timedelta(hours=i) for i in range(24)]
sentiments_history = [random.choices(sentiment_classes, k=20) for _ in range(24)]

# Flatten into a DataFrame
data = []
for t, s_list in zip(timestamps, sentiments_history):
    for s in s_list:
        data.append({"timestamp": t, "sentiment": s})
df = pd.DataFrame(data)

# ----------------------
# Layout
# ----------------------
app.layout = html.Div([
    html.H1("BrandPulse AI - Twitter Sentiment Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.H3("Live Tweet Sentiment Distribution"),
        dcc.Graph(id='pie-chart'),
        dcc.Interval(id='pie-update', interval=5000, n_intervals=0)  # updates every 5 sec
    ], style={'width':'48%', 'display':'inline-block', 'verticalAlign':'top'}),
    
    html.Div([
        html.H3("Sentiment Trend Over 24 Hours"),
        dcc.Graph(id='trend-line'),
        dcc.Interval(id='trend-update', interval=5000, n_intervals=0)  # updates every 5 sec
    ], style={'width':'48%', 'display':'inline-block', 'verticalAlign':'top'})
])

# ----------------------
# Callbacks
# ----------------------
@app.callback(
    Output('pie-chart', 'figure'),
    Input('pie-update', 'n_intervals')
)
def update_pie(n):
    # Simulate new tweet
    new_sentiment = random.choice(sentiment_classes)
    new_time = datetime.now()
    global df
    df = pd.concat([df, pd.DataFrame([{"timestamp": new_time, "sentiment": new_sentiment}])], ignore_index=True)
    
    # Calculate distribution
    counts = df['sentiment'].value_counts().reindex(sentiment_classes, fill_value=0)
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        color_discrete_map={"Positive":"green", "Neutral":"gray", "Negative":"red"},
        title="Live Sentiment Distribution"
    )
    return fig

@app.callback(
    Output('trend-line', 'figure'),
    Input('trend-update', 'n_intervals')
)
def update_trend(n):
    # Group by hour
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly['timestamp'].dt.floor('H')
    trend = df_hourly.groupby(['hour', 'sentiment']).size().reset_index(name='count')
    
    fig = px.line(
        trend,
        x='hour',
        y='count',
        color='sentiment',
        color_discrete_map={"Positive":"green", "Neutral":"gray", "Negative":"red"},
        markers=True,
        title="Sentiment Trend Over Last 24 Hours"
    )
    fig.update_layout(xaxis_title='Time', yaxis_title='Number of Tweets')
    return fig

# ----------------------
# Run Server
# ----------------------
if __name__ == "__main__":
    app.run_server(debug=True)
