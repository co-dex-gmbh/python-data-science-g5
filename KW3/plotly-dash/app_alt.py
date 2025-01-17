from dash import Dash, html, callback, Input, Output
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.SKETCHY])

app.layout = [
    html.Div(children=[
        "Hallo, ich bin eine Dash app"
    ]),
    html.Button("Ich bin ein Button"),
    html.Center(
                children=[
                    dbc.Button("Ich bin ein DBC Button", id="hello_button")
                    ]),
    html.Center(id="main_div"),
]

@callback(
    Output("main_div", "children"),
    
    Input("hello_button", "n_clicks")
)
def react_to_button_click(n_clicks):
    if n_clicks:
        return dbc.Alert("Button wurde geklickt!")
    return




if __name__ == "__main__":
    app.run(debug=True)

