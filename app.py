import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Predictor")

# Load trained pipeline safely
model_path = os.path.join(os.path.dirname(__file__), 'pipe.pkl')
try:
    with open(model_path, 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error(f"❌ Model file not found at {model_path}. Please retrain or upload 'pipe.pkl'.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Input fields
year = st.number_input('Manufacturing Year', min_value=1990, max_value=2025, step=1, value=2015)
kms = st.number_input('Kilometers Driven', min_value=0, step=100, value=50000)
fuel = st.selectbox('Fuel Type', ['Diesel', 'Petrol'])
seller = st.selectbox('Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner'])
mileage = st.number_input('Mileage (km/l)', min_value=0.0, step=0.1, value=15.0)
engine = st.number_input('Engine (CC)', min_value=0.0, step=1.0, value=1200.0)
power = st.number_input('Max Power (bhp)', min_value=0.0, step=0.1, value=75.0)
seats = st.number_input('Seats', min_value=2, max_value=10, step=1, value=5)
brand = st.selectbox(
    'Car Brand',
    ['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Ford', 'Honda', 'Toyota', 'Renault', 'Chevrolet', 'Volkswagen']
)

if st.button('🔮 Predict Price'):
    # Prepare input data
    input_data = pd.DataFrame(
        [[year, kms, fuel, seller, transmission, owner, mileage, engine, power, seats, brand]],
        columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission',
                 'owner', 'mileage', 'engine', 'max_power', 'seats', 'brand']
    )

    try:
        # Predict price
        prediction = pipe.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ₹ {np.round(prediction, 0):,.0f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
