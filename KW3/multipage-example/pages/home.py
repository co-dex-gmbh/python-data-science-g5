import dash


dash.register_page(__name__, path="/")

layout = [
    dash.html.H1("Meine Hauptseite")
]