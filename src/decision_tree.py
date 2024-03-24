import os
import config
import joblib
import ast
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
import data_cleaning
import logging
from tqdm import tqdm
from helper import plot_roc_curve_for_classes
from colorama import Fore
import model_dispatcher

train_roc_auc_list = []
test_roc_auc_list = []
train_classification_reports = []
test_classification_reports = []
def training(fold, model):
    
    
    df = pd.read_csv("../input/folds_csv/oversampled_folds.csv", na_values='unknown')

    ## Here I want to handle missing unkown values 
    df = data_cleaning.handle_missing_values(df)


    train = df[df.kfold != fold].reset_index()
    test = df[df.kfold == fold].reset_index()


    print(Fore.RED + '#'*25)
    print(Fore.LIGHTRED_EX+'### Fold',fold+1)
    print(Fore.LIGHTRED_EX+'### Train size',len(train),'Valid size',len(test))
    print(Fore.RED +'#'*25)
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

    dt = model_dispatcher.models[model]
    
    logging.info(f"Hyperparameters : {dt.get_params()}")
    dt.fit(xtrain, ytrain)
    
    train_preds_prob = dt.predict_proba(xtrain)[:,1]
    test_preds_prob = dt.predict_proba(xvalid)[:,1]

    train_roc_auc = metrics.roc_auc_score(ytrain, train_preds_prob)
    test_roc_auc = metrics.roc_auc_score(yvalid, test_preds_prob)

    train_classification_report = metrics.classification_report(ytrain, dt.predict(xtrain), target_names=['No', 'Yes'])
    test_classification_report = metrics.classification_report(yvalid, dt.predict(xvalid), target_names=['No', 'Yes'])
    logging.info(f"Train ROC AUC: {train_roc_auc}")
    logging.info(f"Test ROC AUC: {test_roc_auc}")


    # save the model
    joblib.dump(
    dt,
    os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

    print(Fore.LIGHTYELLOW_EX + f"\tTrain ROC AUC: {train_roc_auc}")
    print(Fore.LIGHTYELLOW_EX + f"\tTest ROC AUC: {test_roc_auc}")
    
    train_classification_reports.append(train_classification_report)
    test_classification_reports.append(test_classification_report)

    logging.info('Train Classification Report:')
    logging.info(train_classification_report)
    logging.info("Test Classification Report:")
    logging.info(test_classification_report)

    logging.info("#"*25)
    logging.info("\n")
    plot_roc_curve_for_classes(dt, xtrain, ytrain, [0, 1], f'Training ROC Curve for Fold {fold+1}')
    plot_roc_curve_for_classes(dt, xvalid, yvalid, [0, 1], f'Testing ROC Curve for Fold {fold+1}')

    return train_roc_auc, test_roc_auc



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a machine learning model with specified parameters.")

    # # Add arguments
    parser.add_argument('--fold', type=int, required=True, help="Number of folds for cross-validation.")
    parser.add_argument('--model', type=str, required=True, choices=["decision_tree", "lr", "rf"], help="Type of model to train.")
    # Parse arguments
    args = parser.parse_args()
    logging.basicConfig(filename='../logs/checking_with_dispatcher.log', level=logging.INFO)

    # if args.class_weight:
    #     class_weight_dict = ast.literal_eval(args.class_weight)
    # else:
    #     class_weight_dict=None
    # # Run training for each fold
        
    
    for fold in tqdm(range(args.fold)):
        train_roc_auc, test_roc_auc = training(fold, args.model)
        train_roc_auc_list.append(train_roc_auc)
        test_roc_auc_list.append(test_roc_auc)

    logging.info(f"Overall Training ROC Score: {np.mean(train_roc_auc_list)}, Testing ROC Score : {np.mean(test_roc_auc_list)}")
    print(Fore.GREEN +f"Overall Training ROC Score: {np.mean(train_roc_auc_list)}, Testing ROC Score : {np.mean(test_roc_auc_list)}")
  