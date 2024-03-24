import pandas as pd
import numpy as np 
from sklearn import preprocessing 

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def label_encode_categorical_columns(train_df, test_df):

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for column in train_encoded.columns:
        if train_encoded[column].dtype == 'object':
            label_encoder = LabelEncoder()
            train_encoded[column] = label_encoder.fit_transform(train_encoded[column])
            test_encoded[column] = label_encoder.transform(test_encoded[column])

    return train_encoded, test_encoded

def label_encode_categorical_columns_df(df):
    df_encoded = df.copy()

    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            label_encoder =preprocessing.LabelEncoder()
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

    return df_encoded


def handle_missing_values(data):
    
    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['int', 'float']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Impute missing values in numerical columns with median
    imputer = SimpleImputer(strategy='median')
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    # Impute missing values in categorical columns with most frequent value
    imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = imputer.fit_transform(data[categorical_cols])

    return data