import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset.csv")
df.fillna("None", inplace=True)

# Combine Symptoms into List
symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
df['Symptoms'] = df[symptom_cols].values.tolist()

# Create symptom vocabulary
all_symptoms = sorted({sym.strip() for symptoms in df["Symptoms"] for sym in symptoms if sym.strip() != "None"})
symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Function to encode symptoms
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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🩺 Smart Disease Predictor")
st.markdown("Select symptoms and predict the possible disease.")

selected_symptoms = st.multiselect("Type or select symptoms:", all_symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = encode_symptoms(selected_symptoms).reshape(1, -1)
        predicted_disease = label_encoder.inverse_transform(model.predict(input_vector))
        st.success(f"**Predicted Disease:** {predicted_disease[0]}")
