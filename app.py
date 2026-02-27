import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Mobile Price Predictor")

st.title("📱 Mobile Price Prediction App")
st.write("Enter mobile specifications below:")

# Input fields
ram = st.number_input("RAM (MB)", min_value=0)
memory = st.number_input("Internal Memory (GB)", min_value=0)
processor = st.number_input("Processor Speed (GHz)", min_value=0.0)
camera = st.number_input("Primary Camera (MP)", min_value=0)
battery = st.number_input("Battery Power (mAh)", min_value=0)

# Prediction button
if st.button("Predict Price Category"):
    input_data = np.array([[ram, memory, processor, camera, battery]])
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        result = "Low Cost 📉"
    elif prediction == 1:
        result = "Medium Cost 📊"
    elif prediction == 2:
        result = "High Cost 📈"
    else:
        result = "Very High Cost 💎"

    st.success(f"Predicted Price Category: {result}")