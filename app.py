import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("insurance_model.pkl")

st.set_page_config(page_title="Medical Insurance Cost Prediction")

st.title("🏥 Medical Insurance Cost Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=25)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Encode inputs
sex = 1 if gender == "Male" else 0
smoke = 1 if smoker == "Yes" else 0

region_map = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}

region = region_map[region]

if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoke],
        "region": [region]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Insurance Cost: ${prediction[0]:,.2f}")