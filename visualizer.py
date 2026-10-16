import streamlit as st
import pandas as pd

st.set_page_config(page_title="Backpack Doctor App",layout="wide")

st.title("Prediabetes/ Diabetes Prediction")
st.write("This app predicts prediabetes/ diabetes")

st.sidebar.header("User Input Parameters")

#placeholders parameters
def user_input_features():
    age = st.sidebar.number_input(
    "Age in years at screening ",
    min_value=0,
    max_value=60,
    value=30,
    step=1,
    )

    bmi = st.sidebar.number_input(
    "Body Mass Index (kg/m**2)",
    min_value=0.0,
    max_value=25.0,
    value=20.0,
    step=0.1,
    )

    waist = st.sidebar.number_input(
    "Waist Circumference (cm)",
    min_value=60.0,
    max_value=120.0,
    value=90.0,
    step=0.1,
    )

    gender = st.sidebar.selectbox(
    "Gender ",
    ("Male","Female")
    )

    return pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Waist Circunference": [waist],
        "Gender": [gender],

    })

df = user_input_features()


st.subheader("User Input Features")
st.write(df)


