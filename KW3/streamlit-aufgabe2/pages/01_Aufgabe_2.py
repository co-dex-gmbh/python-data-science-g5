import streamlit as st
import pandas as pd
from app import df
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.input_handler import aggregation_mapping

st.set_page_config(layout="wide")

st.header("Aufgabe 2.1")

df["mission_created_date"] = pd.to_datetime(df["mission_created_date"])
df = df.set_index("mission_created_date")

# Aufgabe 2.1.1
selected_cols = st.multiselect(
    "Wähle die Spalten aus, nach denen zu filter möchtest", df.columns)
date_range = st.date_input(
    "Datumsbereich",
    (df.index.min(), df.index.max()),
    min_value=df.index.min(),
    max_value=df.index.max())
# st.write(date_range, type(date_range[0]))
# st.write(min(date_range))
# st.write(selected_cols)


if selected_cols:
    filtered_df = df[(df.index >= pd.to_datetime(min(date_range))) & (
        df.index <= pd.to_datetime(max(date_range)))][selected_cols]
    with st.expander("Dataframe ansehen"):
        filtered_df
    
    # Aufgabe 2.1.2
    # fig = plt.figure()

    # sns.lineplot(filtered_df[selected_cols])
    # st.pyplot(fig)
    with st.expander("Grafik ansehen"):
        aggregation_type = st.selectbox("Aggregationsart", ["Mittelwert", "Summe"], index=0)
        
        aggregation = st.radio("Wähle deine Aggregationsdauer", ("Tag", "Woche", "Monat", "Jahr"))
        window = st.slider("Wähle den Zeitraum für den gleitenden Durchschnitt aus", 
                           min_value=1,
                           max_value=365)
        resolution = aggregation_mapping(aggregation)
            
        if aggregation_type == "Mittelwert":
            filtered_and_resampled = filtered_df[selected_cols].resample(resolution).mean()
        if aggregation_type == "Summe":
            filtered_and_resampled = filtered_df[selected_cols].resample(
                resolution).sum()
            
        filtered_resampled_floating_mean = filtered_and_resampled.rolling(
            window).mean()

        st.line_chart(filtered_resampled_floating_mean)
        
    with st.expander("Grafik mit plotly"):
        fig = px.line(filtered_df[selected_cols])
        st.plotly_chart(fig)
