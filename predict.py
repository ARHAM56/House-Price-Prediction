import joblib
import pandas as pd

# Load trained model and pipeline
model = joblib.load("models/model.pkl")
pipeline = joblib.load("models/pipeline.pkl")


def predict_house_price(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity,
):
    """
    Predict house price using trained model.
    """

    # Create DataFrame
    input_data = pd.DataFrame({
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

    # Apply preprocessing
    transformed_data = pipeline.transform(input_data)

    # Predict
    prediction = model.predict(transformed_data)

    return prediction[0]