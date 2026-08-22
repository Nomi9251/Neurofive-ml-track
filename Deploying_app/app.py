import streamlit as st
import joblib 
import pandas as pd
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent/"model.pkl" 
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

st.title("ML Prediction App")

# User inputs
passenger_id = st.number_input("Passenger ID", min_value=1, value=1)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, value=25.0)
sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, value=0)
parch = st.number_input("Number of Parents/Children", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)

sex = st.selectbox("Sex", ["Female", "Male"])

embarked = st.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)

# Convert Sex into the same format used during training
sex_male = 1 if sex == "Male" else 0

# Convert Embarked into dummy columns
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0


# Creating DataFrame with EXACT same columns as training
input_data = pd.DataFrame({
    "PassengerId": [passenger_id],
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Sex_male": [sex_male],
    "Embarked_Q": [embarked_q],
    "Embarked_S": [embarked_s]
})

 

if st.button("Prediction"):
    

    prediction = model.predict(input_data)
    st.write("Prediction: ", prediction[0])
    "Survived" if prediction[0] == 1 else "Not Survived"