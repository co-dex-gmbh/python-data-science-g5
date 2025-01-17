from dash import Dash, page_registry, page_container
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True,
           external_stylesheets=[dbc.themes.SANDSTONE])


nav_links = [dbc.NavItem(
    dbc.NavLink(page["name"],
                href=page["path"]
                ))
             for page in page_registry.values()]

navbar = dbc.NavbarSimple(
    children=nav_links,
    brand="Analytics Dashboard",
    brand_href="#",
    color="primary",
    dark=True,
)

app.layout = [
    navbar,
    page_container
]


if __name__ == "__main__":
    app.run_server(debug=True)
