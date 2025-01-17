import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__,
                use_pages=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])


nav_links = [
    dbc.NavItem(dbc.NavLink(page["name"],
                            href=page["path"]))
    for page in dash.page_registry.values()
]

navbar = dbc.NavbarSimple(
    children=nav_links
)


app.layout = [
    navbar,
    dash.html.Center(
        [dash.page_container],
        style={"width": "90%"}
    )
]


if __name__ == "__main__":
    app.run(debug=True)
