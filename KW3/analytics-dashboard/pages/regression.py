import dash
from dash import dcc

from dash import html

dash.register_page(__name__)


regression_dropdown = dcc.Dropdown(
    id='regression-dropdown',
    options=[
        {'label': 'Support Vector Regression (SVR)', 'value': 'svr'},
        {'label': 'Decision Tree Regression (DTR)', 'value': 'dtr'},
        {'label': 'Linear Regression', 'value': 'linear'}
    ],
    value='linear'
)

layout = [
    regression_dropdown
]
