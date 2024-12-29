import streamlit as st
import pandas as pd
import os
from utils.model_training import ModelTrainer
from utils.prediction import Predictor

def main():
    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon="❤️",
        layout="wide"
    )

    st.title("❤️ Heart Disease Prediction App")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select a page", ["Train Model", "Make Predictions"])

    if page == "Train Model":
        show_training_page()
    else:
        show_prediction_page()

def show_training_page():
    st.header("Model Training")
    
    if not os.path.exists("Train_Data_csv/heart_disease.csv"):
        st.error("Training data file not found in Train_Data/heart_disease.csv")
        return
    
    if st.button("Train Model"):
        try:
            with st.spinner("Training model... Please wait."):
                trainer = ModelTrainer()
                accuracy, report = trainer.train_model()
                
                st.success(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")
                st.text("Detailed Classification Report:")
                st.text(report)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_prediction_page():
    st.header("Make Predictions")
    
    if not os.path.exists("models/model.pkl"):
        st.error("Model not found. Please train the model first.")
        return
    
    st.info("""
    ### Required Data Format
    Your CSV file should include these columns:
    - Age: Numeric
    - Sex: 'M' or 'F'
    - RestingBP: Numeric (resting blood pressure)
    - Cholesterol: Numeric
    - FastingBS: 0 or 1 (fasting blood sugar)
    - RestingECG: Categorical
    - MaxHR: Numeric (maximum heart rate)
    - ExerciseAngina: 'Y' or 'N'
    - Oldpeak: Numeric
    - ST_Slope: Categorical
    """)

    uploaded_file = st.file_uploader("Upload data for prediction (CSV)", type="csv")
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(data.head())
            
            if st.button("Make Predictions"):
                with st.spinner("Generating predictions..."):
                    predictor = Predictor()
                    results = predictor.predict(data)
                    
                    st.success("Predictions generated successfully!")
                    st.write("Results:")
                    st.dataframe(results)
                    
                    # Download button for results
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results",
                        csv,
                        "heart_disease_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    main()
