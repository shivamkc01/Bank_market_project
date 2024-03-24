import os
import joblib
import pandas as pd
import config
from sklearn import preprocessing
import data_cleaning
def preprocess_data_test(df):
    # Handle missing values if any
    # Example:
    df = data_cleaning.handle_missing_values(df)

    # Label encode categorical columns
    df_encoded = df.copy()
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            label_encoder = preprocessing.LabelEncoder()
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

    # Scale numerical features if needed (using the same scaler as used in training)
    # Example:
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    scaler = preprocessing.StandardScaler()
    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

    return df_encoded


def inference(model_path, test_data_path):
    # Load the trained model
    model = joblib.load(model_path)

    # Load the test data
    test_data = pd.read_csv(test_data_path, sep=";", na_values="unknown")

    # Preprocess the test data
    X_test = preprocess_data_test(test_data)  # Implement the preprocessing steps similar to training data
    print(X_test.shape)
    # Drop the target column if present
    if 'y' in X_test.columns:
        X_test.drop('y', axis=1, inplace=True)

    # Make predictions
    predictions = model.predict(X_test)

    return predictions

if __name__ == "__main__":
    # Define paths to the trained model and test data
    model_path = config.MODEL_OUTPUT + "rf.bin"
    test_data_path = config.TEST_FILE

    # Perform inference
    preds = inference(model_path=model_path, test_data_path=test_data_path)

    # Print or save the predictions
    print(preds)
