import streamlit as st
import pandas as pd


df = pd.read_csv("../berlin-FE/data/BFw_mission_data_daily.csv", skipfooter=1, engine="python")

st.header("Aufgabe 2 Streamlit")


st.markdown(
"""   
2.1 Erstellen Sie ein interaktives Dashboard zur Visualisierung der täglichen Einsatzzahlen

1. Filtern Sie nach Datum und Einsatzarten (z. B. mission_count_fire, mission_count_ems), um die Daten nach verschiedenen Kriterien anzuzeigen.
2. Visualisieren Sie die Anzahl der Einsätze pro Tag/ Woche/ Monat.
3. Zeigen Sie die durchschnittlichen Reaktionszeiten für alle Einsatzarten (z. B. response_time_fire_time_to_first_pump_mean) in einem Liniendiagramm an.
Erstellen Sie interaktive Widgets, mit denen die Benutzer den Zeitraum und die Einsatzarten nach Bedarf anpassen können.
"""
)

