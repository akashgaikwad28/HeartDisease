import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder

def preprocess_data(df):
    # Expected features
    expected_features = [
        'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS',
        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 
        'ChestPainType_ATA'
    ]

    # Check for missing features
    if not all(feature in df.columns for feature in expected_features):
        raise ValueError("Input file must contain all required features.")

    # Keep only the required features
    df = df[expected_features]

    # Initialize label encoder and binary encoder
    label_encoder = LabelEncoder()
    binary_encoder = BinaryEncoder()

    # Apply Binary Encoding on 'ExerciseAngina' and 'Sex'
    df['ExerciseAngina'] = binary_encoder.fit_transform(df['ExerciseAngina'])
    df['Sex'] = binary_encoder.fit_transform(df['Sex'])

    # Apply Label Encoding on 'RestingECG', 'ChestPainType_ATA', and 'ST_Slope'
    df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
    df['ChestPainType_ATA'] = label_encoder.fit_transform(df['ChestPainType_ATA'])
    df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])

    # Numerical features to be scaled
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

    # Initialize the column transformer for scaling numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'
    )

    # Apply the transformations and return the processed data
    processed_data = preprocessor.fit_transform(df)
    
    # Return the DataFrame with appropriate column names
    return pd.DataFrame(processed_data, columns=numerical_features + ['ExerciseAngina', 'Sex', 'RestingECG', 'ST_Slope', 'ChestPainType_ATA'])
