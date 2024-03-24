import os
import ast
import time
import joblib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
import data_cleaning
import logging
from tqdm import tqdm
from helper import plot_roc_curve_for_classes
import model_dispatcher
from colorama import Fore
import config


train_roc_auc_list = []
test_roc_auc_list = []
train_classification_reports = []
test_classification_reports = []
def training(fold, model):
    start = time.time()
    
    df = pd.read_csv(config.OVERSAMPLED_DATA, na_values='unknown')

    ## Here I want to handle missing unkown values 
    df = data_cleaning.handle_missing_values(df)


    train = df[df.kfold != fold].reset_index()
    test = df[df.kfold == fold].reset_index()


    print(Fore.RED+ '#'*25)
    print('### Fold',fold+1)
    print('### Train size',len(train),'Valid size',len(test))
    print('#'*25)
    logging.info("#"*25)
    logging.info(f"### FOLD {fold+1} ###")
    logging.info(f"### Train size , {len(train)}, Valid size , {len(test)}")
    train_enc, test_enc = data_cleaning.label_encode_categorical_columns(train, test)
    

    xtrain = train_enc.drop('y', axis=1).values
    ytrain = train_enc.y.values

    xvalid = test_enc.drop('y', axis=1).values
    yvalid = test_enc.y.values 
    yvalid_series = pd.Series(yvalid)

    scaler = preprocessing.StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xvalid = scaler.transform(xvalid)


    logging.info(f"Model Name : {args.model}")
    model = model_dispatcher.models[model]
    
    logging.info(f"Hyperparameters : {model.get_params()}")
    model.fit(xtrain, ytrain)
    print(xtrain.shape)
    train_preds_prob = model.predict_proba(xtrain)[:,1]
    test_preds_prob = model.predict_proba(xvalid)[:,1]

    train_roc_auc = metrics.roc_auc_score(ytrain, train_preds_prob)
    test_roc_auc = metrics.roc_auc_score(yvalid, test_preds_prob)

    train_classification_report = metrics.classification_report(ytrain, model.predict(xtrain), target_names=['No', 'Yes'])
    test_classification_report = metrics.classification_report(yvalid, model.predict(xvalid), target_names=['No', 'Yes'])
    logging.info(f"Train ROC AUC: {train_roc_auc}")
    logging.info(f"Test ROC AUC: {test_roc_auc}")

    print(Fore.GREEN+f"Train ROC AUC: {train_roc_auc}")
    print(Fore.GREEN+f"Test ROC AUC: {test_roc_auc}")
    
    train_classification_reports.append(train_classification_report)
    test_classification_reports.append(test_classification_report)

    logging.info('Train Classification Report:')
    logging.info(train_classification_report)
    logging.info("Test Classification Report:")
    logging.info(test_classification_report)

    logging.info("#"*25)
    logging.info("\n")
    plot_roc_curve_for_classes(model, xtrain, ytrain, [0, 1], f'Training ROC Curve for Fold {fold+1}')
    plot_roc_curve_for_classes(model, xvalid, yvalid, [0, 1], f'Testing ROC Curve for Fold {fold+1}')

    end_time = time.time()
    fold_time = end_time - start
    logging.info(f"Time taken for fold {fold+1}: {fold_time} seconds")
    

    return train_roc_auc, test_roc_auc, model



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a machine learning model with specified parameters.")

    # Add arguments
    parser.add_argument('--fold', type=int, required=True, help="Number of folds for cross-validation.")
    parser.add_argument('--model', type=str, required=True, choices=["decision_tree", "rf", "lr"], help="Type of model to train.")
    
    # Parse arguments
    args = parser.parse_args()
    logging.basicConfig(filename=config.LOGS_FILE+"best_models.log", level=logging.INFO)

    train_roc_auc_avg = []
    test_roc_auc_avg = []
    best_model = None

    for fold in tqdm(range(args.fold)):
        train_roc_auc, test_roc_auc, model = training(fold, args.model)
        train_roc_auc_avg.append(train_roc_auc)
        test_roc_auc_avg.append(test_roc_auc)
        if best_model is None or test_roc_auc > max(test_roc_auc_avg):
            best_model = model

    logging.info(f"Overall Training ROC Score: {np.mean(train_roc_auc_avg)}, Testing ROC Score : {np.mean(test_roc_auc_avg)}")
    print(f"Overall Training ROC Score: {np.mean(train_roc_auc_avg)}, Testing ROC Score : {np.mean(test_roc_auc_avg)}")
    
    joblib.dump(
        model,
        os.path.join(config.MODEL_OUTPUT, f"{args.model}.bin")
    )
    print(Fore.LIGHTYELLOW_EX + f"Successfully done the training using {args.model}")