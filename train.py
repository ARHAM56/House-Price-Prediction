import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"


def build_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline


def train():

    os.makedirs("models", exist_ok=True)

    housing = housing = pd.read_csv("data/housing.csv")

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3., 4.5, 6., np.inf],
        labels=[1,2,3,4,5]
    )

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    for train_index, test_index in split.split(housing, housing["income_cat"]):

        housing.iloc[test_index].drop(
            "income_cat",
            axis=1
        ).to_csv("data/input.csv", index=False)

        housing = housing.iloc[train_index].drop(
            "income_cat",
            axis=1
        )

    labels = housing["median_house_value"].copy()

    features = housing.drop("median_house_value", axis=1)

    num_attribs = features.drop(
        "ocean_proximity",
        axis=1
    ).columns.tolist()

    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(
        num_attribs,
        cat_attribs
    )

    prepared_data = pipeline.fit_transform(features)

    model = RandomForestRegressor(random_state=42)

    model.fit(prepared_data, labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained successfully!")
    print("Model saved in models/model.pkl")
    print("Pipeline saved in models/pipeline.pkl")


if __name__ == "__main__":
    train()