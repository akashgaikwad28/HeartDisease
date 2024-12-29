import os
import pickle
import pandas as pd
from .data_preprocessing import DataPreprocessor

class Predictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_model()

    def load_model(self):
        """Load the trained model and preprocessor"""
        if not os.path.exists("models/model.pkl") or not os.path.exists("models/preprocessor.pkl"):
            raise FileNotFoundError("Model or preprocessor not found. Please train the model first.")

        try:
            self.preprocessor = DataPreprocessor.load("models/preprocessor.pkl")
            with open("models/model.pkl", "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict(self, df):
        """Make predictions on the input data"""
        try:
            X = self.preprocessor.preprocess_data(df, is_training=False)
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            results = df.copy()
            results['Heart_Disease_Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
            results['Risk_Probability'] = [f"{prob[1] * 100:.2f}%" for prob in probabilities]
            
            return results
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
