import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

    # One-hot encoding for categorical variables
    categorical_features = ['Sex', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'ChestPainType_ATA']
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Apply transformations
    processed_data = preprocessor.fit_transform(df)
    return pd.DataFrame(processed_data, columns=numerical_features + preprocessor.named_transformers_['cat'].get_feature_names_out())
