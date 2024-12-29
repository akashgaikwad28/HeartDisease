# Heart Disease Prediction using Machine Learning

This project is a heart disease prediction system built using machine learning, specifically the **GradientBoostingClassifier**. The application allows users to upload their own dataset in CSV format, preprocesses it, and predicts whether a patient has heart disease (Yes/No) based on various features like age, sex, blood pressure, cholesterol, and more. The output is saved as a CSV file, which includes the original features along with a new feature indicating the prediction result.

## Features

- **Data Preprocessing**: Cleans and transforms raw CSV data, handling missing values and encoding categorical features.
- **Heart Disease Prediction**: Uses a trained **GradientBoostingClassifier** model to predict the presence of heart disease.
- **CSV Output**: The resulting predictions are added as a new column in the original dataset, with `Yes`/`No` indicating the prediction.
- **User Upload**: Allows users to upload their own CSV dataset.
- **Streamlit Interface**: Easy-to-use web app built using **Streamlit** for interaction.

## Technologies Used

- **Python**: The core language for the project.
- **Streamlit**: Framework to build the interactive web app.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For model training and machine learning algorithms.
- **XGBoost**: For enhanced prediction performance (used optionally).
- **Numpy**: For numerical operations.
- **Matplotlib/Seaborn**: For data visualization.

## How It Works

1. **Data Upload**: Users upload a CSV file with patient data containing the following columns:
   - `Age`, `Sex`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`
   - `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, `ChestPainType_ATA`
   
2. **Preprocessing**:
   - Missing data is handled.
   - Categorical features are encoded.
   - Numerical features are scaled using standardization.

3. **Prediction**: The data is fed into a **GradientBoostingClassifier** to predict the likelihood of heart disease.

4. **Output**: A new column `HeartDisease` is added to the uploaded dataset indicating `Yes` or `No`.

5. **Download Result**: Users can download the modified CSV file with predictions.

## Installation

To run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/heart_disease_prediction.git
cd heart_disease_prediction
