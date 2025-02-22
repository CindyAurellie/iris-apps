import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load("iris_model.pkl")

# Load Iris dataset details
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
st.write("Enter feature values to predict the flower type.")

# Input fields for user
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict button
if st.button("Predict"):
    # Convert input into numpy array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(features)
    predicted_class = target_names[prediction[0]]
    
    # Display result
    st.success(f"Predicted Flower Type: **{predicted_class}** 🌼")
