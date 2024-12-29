import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pickle

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {
            'RestingECG': LabelEncoder(),
            'ST_Slope': LabelEncoder(),
            'ChestPainType': LabelEncoder()  
        }
        self.scaler = StandardScaler()
        self.is_fitted = False

    def preprocess_data(self, df, is_training=True):
        """
        Preprocess the data for model training or prediction.
        """
        required_features = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
        ]
        if is_training:
            required_features.append('HeartDisease')

        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required columns: {missing_features}")

        df = df.copy()
        df = df.dropna(subset=required_features)

        # Encode categorical features
        df['Sex'] = df['Sex'].map({'M': 1, 'F': 0}).fillna(-1)
        df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0}).fillna(-1)

        # Handle categorical encodings
        for feature, encoder in self.label_encoders.items():
            if is_training:
                df[feature] = encoder.fit_transform(df[feature].fillna('unknown'))
            else:
                # Handle unseen categories in prediction
                df[feature] = df[feature].fillna('unknown')
                unique_values = encoder.classes_
                df[feature] = df[feature].apply(lambda x: x if x in unique_values else 'unknown')
                df[feature] = encoder.transform(df[feature])

        # Standardize numerical features
        numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        if is_training:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features].fillna(0))
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Preprocessor must be fitted before transform")
            df[numerical_features] = self.scaler.transform(df[numerical_features].fillna(0))

        if is_training:
            X = df.drop(columns=['HeartDisease'])
            y = df['HeartDisease']
            return X, y
        return df

    def save(self, path):
        """Save the preprocessor state"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load the preprocessor state"""
        with open(path, 'rb') as f:
            return pickle.load(f)
