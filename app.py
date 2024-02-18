import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Load or preprocess your data to obtain features and scaler
# Example: Assuming you have a CSV file 'your_data.csv'
data = pd.read_csv('sfari_genes.csv')
features = data.drop(columns=['syndromic', 'gene-symbol'])
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(features)

# Load the model
model = XGBClassifier()
# Assuming your model is trained on the entire dataset before deployment
model.fit(X, data['syndromic'])

# Function to preprocess input data
def preprocess_data(input_data):
    input_data = pd.DataFrame([input_data], columns=features.columns)
    input_data = pd.get_dummies(data=input_data, columns=['chromosome', 'genetic-category', 'gene-score'])
    
    # Convert any string inputs to float
    for column in input_data.columns:
        if input_data[column].dtype == 'O':  # 'O' stands for object (usually string)
            input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
    
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

st.title("Gene Syndromic Prediction")

# Get user input for gene symbol
gene_symbol = st.text_input("Enter Gene Symbol:")

if st.button("Predict"):
    try:
        # Get the corresponding row from the dataset for the entered gene symbol
        input_data_row = data[data['gene-symbol'] == gene_symbol].drop(['syndromic', 'gene-symbol'], axis=1).iloc[0]
        
        # Preprocess the input data for prediction
        input_data_scaled = preprocess_data(input_data_row)
        
        # Make the prediction
        prediction = model.predict(input_data_scaled.reshape(1, -1))
        
        result = "Syndromic" if prediction == 1 else "Non-Syndromic"
        st.write(f"Prediction for Gene Symbol '{gene_symbol}': {result}")
    except IndexError:
        st.write(f"Gene Symbol '{gene_symbol}' not found in the dataset.")
    except ValueError as e:
        st.write(f"Error: {e}. Please make sure the input is valid.")
