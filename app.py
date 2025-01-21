import streamlit as st
import pickle
import pandas as pd

from pyngrok import ngrok
import threading
import os

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app title
st.title("Predict with Machine Learning Model")

# Input fields for features
st.write("Enter the values for the following features:")

purchase_frequency = st.number_input("Purchase Frequency", value=0.0, step=0.1)
average_order_value = st.number_input("Average Order Value", value=0.0, step=1.0)
churn_probability = st.number_input("Churn Probability", value=0.0, step=0.01)
time_between_purchases = st.number_input("Time Between Purchases (days)", value=0, step=1)

# Predict button
if st.button("Predict"):
    # Prepare the input data as a DataFrame
    try:
        # Step 1: Input data
        input_data = pd.DataFrame([{
            "Purchase_Frequency": purchase_frequency,
            "Average_Order_Value": average_order_value,
            "Churn_Probability": churn_probability,
            "Time_Between_Purchases": time_between_purchases,
        }])

        # Step 4: Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"The predicted value is: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def run_streamlit():
    os.system('streamlit run app.py --server.port 8501')

run_streamlit()

#public_url = ngrok.connect(addr='8501')
#print(f'Streamlit app is live at: {public_url}')