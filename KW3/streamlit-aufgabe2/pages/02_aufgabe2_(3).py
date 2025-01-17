from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
from app import df
import seaborn as sns
import matplotlib.pyplot as plt


selected_cols = st.multiselect("Wähle die Eingangsvariablen aus", df.columns)

if selected_cols:
    relevant_cols = df[selected_cols +
                       ["response_time_fire_time_to_first_pump_mean"]]

    X_train, X_test, y_train, y_test = train_test_split(
        relevant_cols.drop(
            "response_time_fire_time_to_first_pump_mean", axis=1),
        relevant_cols["response_time_fire_time_to_first_pump_mean"],
        test_size=0.2,
        random_state=73
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    fig = plt.figure()
    sns.scatterplot(
        relevant_cols, 
        x=relevant_cols.iloc[:, 0], 
        y="response_time_fire_time_to_first_pump_mean")
    sns.lineplot(relevant_cols, 
                 x=relevant_cols.iloc[:, 0], 
                 y=model.predict(relevant_cols.drop("response_time_fire_time_to_first_pump_mean", 
                                                    axis=1)))
    st.pyplot(fig)

    st.write("Model R^2")
    st.write(model.score(X_test, y_test))
