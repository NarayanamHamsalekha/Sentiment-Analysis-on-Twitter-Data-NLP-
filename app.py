# app.py
import dash
from dash import html, dcc
import plotly.express as px

app = dash.Dash(__name__)
server = app.server  # Needed for deployment

# Example Pie chart
fig = px.pie(names=["Positive","Neutral","Negative"], values=[50,30,20])

app.layout = html.Div([
    html.H1("BrandPulse AI - Sentiment Analysis"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)
