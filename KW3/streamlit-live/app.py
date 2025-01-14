import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Authentifizierung
def check_password():
    def password_entered():
        # username, password = st.secrets.credentials["username"]
        if st.session_state["password"] == st.secrets.credentials["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Lösche das Passwort aus dem Zustand
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Passwort", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Passwort", type="password", on_change=password_entered, key="password")
        st.error("Passwort falsch")
        return False
    else:
        return True

if check_password():
    # Einführung in Streamlit
    st.title('Streamlit Schulung')

    # Seite 1: Einführung
    st.header('1. Einführung in Streamlit')
    st.write("Streamlit ermöglicht es, interaktive Web-Apps in Python zu erstellen. Dies ist eine Einführung in die grundlegenden Funktionen.")

    # Seite 2: Struktur
    st.header('2. Struktur einer Streamlit-App')
    st.write("Streamlit nutzt eine einfache Struktur: Der Python-Code wird direkt in eine Web-Anwendung übersetzt.")
    st.subheader('Beispiel für eine einfache App:')
    st.code("""
    import streamlit as st
    st.title("Streamlit App")
    """)

    # Seite 3: Visualisierungen
    st.header('3. Visualisierungen in Streamlit')
    st.write("Streamlit ermöglicht die einfache Einbindung von Visualisierungen.")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

    # Seite 4: Layout
    st.header('4. Layout und Design')
    st.write("Streamlit ermöglicht das Erstellen von benutzerdefinierten Layouts mit Spalten und Sidebars.")
    sidebar_input = st.sidebar.text_input('Gib etwas ein:')
    st.write(f'Du hast folgendes eingegeben: {sidebar_input}')

    # Seite 5: Umgang mit Daten
    st.header('5. Umgang mit Daten')
    data = pd.DataFrame({'Name': ['A', 'B', 'C'], 'Alter': [23, 34, 45]})
    st.write(data)

    # Seite 6: Interaktivität
    st.header('6. Interaktive Elemente')
    if st.button('Klicke mich'):
        st.write('Button wurde geklickt!')

    value = st.slider('Wähle eine Zahl', 0, 100)
    st.write(f'Du hast {value} ausgewählt.')

    # Seite 7: Caching
    @st.cache_data
    def expensive_computation():
        return pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])

    df = expensive_computation()
    st.write(df)

    # Seite 8: Multipage App
    st.header('8. Multipage App')
    page = st.selectbox('Wähle eine Seite', ['Seite 1', 'Seite 2'])
    if page == 'Seite 1':
        st.write('Dies ist die erste Seite.')
    elif page == 'Seite 2':
        st.write('Dies ist die zweite Seite.')

    # Seite 9: Secrets und Authentication
    st.header('9. Secrets und Authentication')
    st.write("Geheime Schlüssel können über die .streamlit/secrets.toml-Datei geladen werden.")
    api_key = st.secrets["api_key"]
    st.write(f'API-Schlüssel: {api_key}')

    # Ausblick
    st.header('10. Ausblick')
    st.write('Streamlit bietet noch viele weitere Funktionen. In einer Produktionsumgebung können Apps auf Plattformen wie Heroku oder Streamlit Sharing gehostet werden.')

