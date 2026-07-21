# 🏠 California House Price Prediction

A Machine Learning web application built with **Python, Scikit-learn, and Streamlit** to predict California house prices based on housing features.

---

## 📸 Demo

![App Screenshot](assets/logo.png)

---

## 🚀 Features

- 🏠 Predict California house prices
- 🤖 Random Forest Regression model
- 📊 Data preprocessing using Scikit-learn Pipeline
- 🎨 Interactive Streamlit UI
- 📈 Clean project structure
- ⚡ Fast predictions
- 🧩 Modular code (Train, Predict, UI separated)

---

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Plotly

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── app.py                 # Streamlit frontend
├── train.py               # Train ML model
├── predict.py             # Prediction logic
├── requirements.txt
├── README.md
│
├── data/
│   ├── housing.csv
│   ├── input.csv
│   └── output.csv
│
├── models/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── assets/
│   ├── logo.png
│   └── style.css
│
└── venv/
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ARHAM56/House-Price-Prediction.git
```

## 2. Go to the project folder

```bash
cd House-Price-Prediction
```

## 3. Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Train the Model

```bash
python train.py
```

This creates:

- model.pkl
- pipeline.pkl

---

## 6. Run the Streamlit App

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 📊 Input Features

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

---

# 🧠 Machine Learning Model

Algorithm Used:

- Random Forest Regressor

Preprocessing:

- Missing Value Imputation
- Standard Scaling
- One-Hot Encoding
- Scikit-learn Pipeline

---

# 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
plotly
matplotlib
```

---

# 📸 Screenshots

Add screenshots here after uploading images.

Example:

```
assets/home.png
assets/predict.png
```

---

# 📈 Future Improvements

- Deep Learning Model
- XGBoost
- LightGBM
- Feature Importance Chart
- Prediction History
- Download Prediction Report
- Deploy on Streamlit Cloud
- User Authentication

---

# 👨‍💻 Author

**Arham Aziz**

AI & Machine Learning Engineering Student

GitHub:
https://github.com/ARHAM56

LinkedIn:
https://www.linkedin.com/in/arham-aziz-aa48ba41b/

---

# ⭐ If you like this project

Please give this repository a ⭐ on GitHub.
