import streamlit as st
import numpy as np
from src.predict import predict_transaction

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to check if it's fraudulent.")

# Collect all 30 features (V1-V28 + Time + Amount)
features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", -20.0, 20.0, 0.0)
    features.append(val)

time_val = st.number_input("Time (scaled)", -100.0, 1000.0, 0.0)
features.append(time_val)

amount_val = st.number_input("Amount (scaled)", -100.0, 100.0, 0.0)
features.append(amount_val)

if st.button("Predict"):
    prediction, prob = predict_transaction(features)

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction. (Confidence: {prob:.2f})")
