import dash
from dash import dash_table, callback, Output, Input, State, html
import pandas as pd

dash.register_page(__name__)


initial_df = pd.DataFrame()

layout = [
    dash_table.DataTable(
        id="editable-table",
        columns=[
            {"name": "x", "id": "x", "type": "numeric", "editable": True},
            {"name": "y", "id": "y", "type": "numeric", "editable": True}
        ],
        data=initial_df.to_dict("records"),
        editable=True,
        row_deletable=True,
        row_selectable="multi",
    ),

    html.Button("Neue Zeile hinzufügen", id="add-row-btn", n_clicks=0),
    html.Button("Daten aktualisieren", id="update-graph-btn", n_clicks=0)
]


@callback(
    Output('editable-table', 'data'),
    Input('add-row-btn', 'n_clicks'),
    State('editable-table', 'data'),
    State('editable-table', 'columns')
)
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({col['id']: None for col in columns})
    return rows


