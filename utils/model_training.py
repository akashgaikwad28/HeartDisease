import os
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from .data_preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

    def train_model(self):
        """Train the model using the fixed training data path"""
        data_path = "Train_Data/heart_disease.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found at {data_path}")

        try:
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)

            # Load and preprocess data
            df = pd.read_csv(data_path)
            X, y = self.preprocessor.preprocess_data(df, is_training=True)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train the model
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # Save preprocessor and model
            self.preprocessor.save("models/preprocessor.pkl")
            with open("models/model.pkl", "wb") as f:
                pickle.dump(self.model, f)

            return accuracy, report

        except Exception as e:
            raise Exception(f"Error during model training: {str(e)}")
