import streamlit as st
import pandas as pd
from preprocess import preprocess_data
from model import load_model, make_predictions

st.title("Heart Disease Prediction")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    user_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(user_data)

    # Preprocess the data
    st.write("Preprocessing data...")
    try:
        processed_data = preprocess_data(user_data)
        st.write("Data after preprocessing:")
        st.dataframe(processed_data)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")

    # Load pre-trained model
    model = load_model("models/gradient_boosting_model.pkl")

    # Make predictions
    if st.button("Predict"):
        st.write("Predicting...")
        predictions = make_predictions(model, processed_data)
        user_data['HeartDisease_Prediction'] = predictions
        user_data['HeartDisease_Prediction'] = user_data['HeartDisease_Prediction'].map({1: "Yes", 0: "No"})
        
        st.write("Predictions:")
        st.dataframe(user_data)

        # Download results
        csv = user_data.to_csv(index=False)
        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="heart_disease_predictions.csv",
            mime="text/csv"
        )
