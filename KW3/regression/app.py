from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

app = Dash(__name__)

# Sample DataFrame
df = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [1, 4, 9, 16, 25]
})

app.layout = html.Div([
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'SVR', 'value': 'SVR'},
            {'label': 'Linear Regression', 'value': 'LR'},
            {'label': 'Decision Tree Classifier', 'value': 'DTC'}
        ],
        value='LR'
    ),
    dcc.Input(id='input-x', type='number', placeholder='X value'),
    dcc.Input(id='input-y', type='number', placeholder='Y value'),
    html.Button('Add Point', id='add-point-button', n_clicks=0),
    html.Button('Fit Model', id='fit-model-button', n_clicks=0),
    dcc.Graph(id='model-graph'),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Output('model-graph', 'figure'),
    [Input('fit-model-button', 'n_clicks')],
    [State('model-dropdown', 'value')]
)
def fit_model(n_clicks, model_name):
    if n_clicks > 0:
        X = df[['X']]
        y = df['Y']
        if model_name == 'SVR':
            model = SVR()
        elif model_name == 'LR':
            model = LinearRegression()
        elif model_name == 'DTC':
            model = DecisionTreeClassifier()
        model.fit(X, y)
        
        # Vorhersagen für den Graphen
        X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_pred)
        
        fig = px.scatter(df, x='X', y='Y', title='Model Visualization')
        fig.add_scatter(x=X_pred.flatten(), y=y_pred, mode='lines', name='Model Prediction')
        
        return f'Model {model_name} fitted successfully!', fig
    return '', {}

@app.callback(
    Output('output', 'children'),
    [Input('add-point-button', 'n_clicks')],
    [State('input-x', 'value'), State('input-y', 'value')]
)
def add_point(n_clicks, x, y):
    if n_clicks > 0 and x is not None and y is not None:
        global df
        df = df.append({'X': x, 'Y': y}, ignore_index=True)
        return f'Point ({x}, {y}) added to DataFrame!'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)