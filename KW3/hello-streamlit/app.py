import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from app import df

# st.write("Willkommen beim Feuerwehr-Dashboard!")

"Willkommen beim Feuerwehr-Dashboard!"

st.header("Ich bin eine Übeschrift")
st.subheader("Ich bin ein Subheader")

# st.markdown()

@st.cache_data()
def create_dataframe():
    return pd.DataFrame({"Name": ["Chris", "Johannes", "Chris"]})

df = create_dataframe()
st.session_state.df = df

# Ein paar Varianten zum Anzeigen eines Data Frames
# Wir packen alle in eine Zeile
cols = st.columns(3)
with cols[0]:
    st.dataframe(df)
with cols[1]:
    df
    result = st.slider("Mein Slider", min_value=0, max_value=10)
    result
with cols[2]:
    st.write(df)


st.sidebar.text_input("Hier kann etwas geschrieben werden")

button_result = st.button("Ich bin ein Button")

# st.session_state.button_result = button_result
button_result

# "Session"
# st.session_state.button_result

# Grafiken
fig = plt.figure()
sns.histplot(df["Name"])
st.pyplot(fig)

sns.histplot(df["Name"])
st.pyplot(fig)

# wenn neue daten hinzugefügt wurden
# oder einmal am Tag
st.cache_data.clear()
