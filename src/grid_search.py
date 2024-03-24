import pandas as pd
import numpy as np 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import model_selection
import data_cleaning
from sklearn import metrics

def perform_grid_search(xtrain, ytrain):
    dt = ensemble.RandomForestClassifier()

    param_grid = {
    'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt'],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }

    # Create a RandomizedSearchCV object
    grid_search = RandomizedSearchCV(estimator=dt,
                                     param_distributions=param_grid, 
                                     cv=10, 
                                     scoring='roc_auc',
                                     n_jobs=-1,
                                     verbose=10)

    # Perform grid search
    grid_search.fit(xtrain, ytrain)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    

    return best_params

def label_encode_categorical_columns(df):
    df_encoded = df.copy()

    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            label_encoder =preprocessing.LabelEncoder()
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

    return df_encoded

if __name__ == "__main__":
    # Load your dataset and preprocess it
    # For example:
    df = pd.read_csv("../input/folds_csv/oversampled_folds.csv", na_values='unknown')
    df = data_cleaning.handle_missing_values(df)
    
    df_encoded = label_encode_categorical_columns(df)
    X = df_encoded.drop('y', axis=1).values
    y = df_encoded['y'].values
    
    xtrain, xvalid, ytrain, yvalid = model_selection.train_test_split(X, y , test_size=0.20, random_state=42)
    scaler = preprocessing.StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xvalid)

    
    best_params = perform_grid_search(xtrain_scaled, ytrain)
    print("Best parameters:", best_params)
    # Perform grid search to get the best ROC AUC score
    dt = ensemble.RandomForestClassifier(**best_params)  # Initialize DecisionTreeClassifier with best parameters
    dt.fit(xtrain_scaled, ytrain)  # Fit the model
    
    # Predict probabilities for the training and validation sets
    y_train_pred_proba = dt.predict_proba(xtrain_scaled)[:, 1]
    y_valid_pred_proba = dt.predict_proba(xtest_scaled)[:, 1]
    
    # Calculate ROC AUC scores
    train_roc_auc = metrics.roc_auc_score(ytrain, y_train_pred_proba)
    valid_roc_auc = metrics.roc_auc_score(yvalid, y_valid_pred_proba)
    
    print("Training ROC AUC Score:", train_roc_auc)
    print("Validation ROC AUC Score:", valid_roc_auc)
