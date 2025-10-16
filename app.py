import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🚗 Car Price Predictor")

# 1️⃣ Try loading the trained pipeline
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    pipe = None  # Prevent NameError

# 2️⃣ Input fields
year = st.number_input('Manufacturing Year', min_value=1990, max_value=2025, step=1)
kms = st.number_input('Kilometers Driven', min_value=0, step=100)
fuel = st.selectbox('Fuel Type', ['Diesel', 'Petrol'])
seller = st.selectbox('Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner'])
mileage = st.number_input('Mileage (km/l)', min_value=0.0)
engine = st.number_input('Engine (CC)', min_value=0.0)
power = st.number_input('Max Power (bhp)', min_value=0.0)
seats = st.number_input('Seats', min_value=2, max_value=10, step=1)
brand = st.selectbox('Car Brand', ['Maruti','Hyundai','Mahindra','Tata','Ford','Honda','Toyota','Renault','Chevrolet','Volkswagen'])

# 3️⃣ Prediction
if st.button('🔮 Predict Price'):
    if pipe is None:
        st.error("Model not loaded. Please retrain first!")
    else:
        input_data = pd.DataFrame(
            [[year, kms, fuel, seller, transmission, owner, mileage, engine, power, seats, brand]],
            columns=['year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats','brand']
        )
        prediction = pipe.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ₹ {np.round(prediction, 2)}")
