import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../input/oversampled_dataset.csv", na_values='unknown')

    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=10, shuffle=True)
    y = df.y 

    for f, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_idx, "kfold"] = f

    df.to_csv('../input/folds_csv/oversampled_folds.csv', index=False)