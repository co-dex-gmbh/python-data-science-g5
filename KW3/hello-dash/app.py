import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

app.layout = html.Div(children=[
    html.Center(children=[
        html.H1(children='Hello Dash'),    
        html.Div(children='''
            Dash: A web application framework for Python.
        '''),
    ]),
    html.Button(
        children=[
            html.A("Das ist ein Link")   
        ]
    ),
    dbc.Button("Das ist ein Button in dbc"),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3, 4, 5], 'y': [4, 1, 3, 5, 2], 'type': 'line', 'name': 'SF'},
                {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 5, 3, 1], 'type': 'bar', 'name': 'NYC'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
]
)

if __name__ == '__main__':
    app.run_server(debug=True)