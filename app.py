import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained pipeline safely
with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

st.title("🚗 Car Price Predictor")

# Input fields
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
brand = st.selectbox(
    'Car Brand',
    ['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Ford', 'Honda', 'Toyota', 'Renault', 'Chevrolet', 'Volkswagen']
)

if st.button('🔮 Predict Price'):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame(
        [[year, kms, fuel, seller, transmission, owner, mileage, engine, power, seats, brand]],
        columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission',
                 'owner', 'mileage', 'engine', 'max_power', 'seats', 'brand']
    )

    # Predict price
    prediction = pipe.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ₹ {np.round(prediction, 2)}")
