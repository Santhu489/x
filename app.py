import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Load or preprocess your data to obtain X
# Example: Assuming you have a CSV file 'your_data.csv'
data = pd.read_csv('your_data.csv')
features = data.drop(columns='syndromic')
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(features)

# Load the model
model = XGBClassifier()
# Assuming your model is trained on the entire dataset before deployment
model.fit(X, data['syndromic'])

# Function to preprocess input data
def preprocess_data(input_data):
    input_data = pd.DataFrame(input_data, columns=features.columns)
    input_data = pd.get_dummies(data=input_data, columns=['chromosome', 'genetic-category', 'gene-score'])
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

st.title("Autism Prediction")

# Get user input
input_data = {}
for feature in features.columns:
    input_data[feature] = st.number_input(f"{feature}:", step=0.1)

if st.button("Predict"):
    input_data_list = list(input_data.values())
    input_data_scaled = preprocess_data([input_data_list])
    prediction = model.predict(input_data_scaled)
    result = "Autism" if prediction == 1 else "Not Autism"
    st.write(f"Prediction: {result}")
