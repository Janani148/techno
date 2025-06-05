import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.fillna("None", inplace=True)
    symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
    df["Symptoms"] = df[symptom_cols].values.tolist()
    df["Symptoms"] = df["Symptoms"].apply(lambda x: [sym.strip() for sym in x if sym.strip() != "None"])
    return df

df = load_data()

# Extract symptoms and diseases
all_symptoms = sorted({sym for symptoms in df["Symptoms"] for sym in symptoms})
symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Encode symptoms
def encode_symptoms(symptom_list):
    vector = np.zeros(len(all_symptoms), dtype=int)
    for symptom in symptom_list:
        if symptom in symptom_index:
            vector[symptom_index[symptom]] = 1
    return vector

# Prepare training data
X = np.array([encode_symptoms(s) for s in df["Symptoms"]])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Disease"])

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# UI layout
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 Disease Predictor Based on Symptoms")
st.markdown("Select your symptoms from the dropdown below.")

# User input
user_symptoms = st.multiselect("Select Symptoms", options=all_symptoms)

if st.button("Predict Disease"):
    if not user_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        input_vector = encode_symptoms(user_symptoms).reshape(1, -1)
        prediction = model.predict(input_vector)
        disease_name = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🧬 **Predicted Disease:** `{disease_name}`")

