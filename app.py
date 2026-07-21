# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ----------------------------
# Load Model & Pipeline
# ----------------------------
model = joblib.load("models/model.pkl")
pipeline = joblib.load("models/pipeline.pkl")

# ----------------------------
# Title
# ----------------------------
st.title("🏠 California House Price Prediction")
st.write("Predict house prices using a Machine Learning model.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.image("assets/logo.png")
st.sidebar.header("Enter House Details")

longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -122.0)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0)

housing_median_age = st.sidebar.slider(
    "Housing Median Age",
    1,
    52,
    20
)

total_rooms = st.sidebar.number_input(
    "Total Rooms",
    min_value=1,
    value=2000
)

total_bedrooms = st.sidebar.number_input(
    "Total Bedrooms",
    min_value=1,
    value=400
)

population = st.sidebar.number_input(
    "Population",
    min_value=1,
    value=1000
)

households = st.sidebar.number_input(
    "Households",
    min_value=1,
    value=350
)

median_income = st.sidebar.number_input(
    "Median Income",
    min_value=0.1,
    value=3.5
)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

# ----------------------------
# Input DataFrame
# ----------------------------
input_df = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

# ----------------------------
# Predict
# ----------------------------
if st.button("🔍 Predict House Price"):

    transformed = pipeline.transform(input_df)

    prediction = model.predict(transformed)[0]

    st.success("Prediction Completed!")

    st.metric(
        label="Estimated House Price",
        value=f"${prediction:,.2f}"
    )

    st.subheader("Input Summary")

    st.dataframe(input_df)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Scikit-Learn")