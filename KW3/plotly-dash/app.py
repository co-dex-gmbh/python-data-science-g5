from dash import Dash, html, dcc, callback, Input, State, Output, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd



df = pd.DataFrame({"X": [1,2,3], "Y": [2, 1, 2]})
app = Dash(external_stylesheets=[dbc.themes.SLATE])


checklist_card = dbc.Card([
    dbc.Checklist(
        ['New York City', 'Montréal', 'San Francisco'],
        ['New York City', 'Montréal'],
        # style=""
    )],
    style={"width": "150px"}
)


graph = dcc.Graph(id="mein_graph",
                  figure=px.scatter(df, "X", "Y")
                  )


input_field = dbc.Input(id="eingabe_neuer_wert")

app.layout = [
    html.Center([
        html.H1("Feuerwehreinsätze in Berlin"),
        # checklist_card,
        input_field,
        graph
    ]),
]


@app.callback(
    Output("mein_graph", "figure"),
    State("eingabe_neuer_wert", "value"),
    Input("eingabe_neuer_wert", "n_submit"),
    prevent_inital_call=True
)
def update_graph(val, n_submit):
    global df
    print(val)

    try:
        val = float(val)
    except:
        return no_update
    
    new_row = pd.DataFrame({"X": [val], "Y": [val**2]})
    df = pd.concat([df, new_row])
    return px.scatter(df, "X", "Y")

if __name__ == "__main__":
    app.run(debug=True, port=5001)



