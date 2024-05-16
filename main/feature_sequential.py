from sklearnex import patch_sklearn   # noreorder

patch_sklearn()

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from mlxtend.feature_selection import SequentialFeatureSelector
from rich import print
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import joblib


def sequential_feature_selection(X_train, y_train, save=False, save_path=None):
    print('[purple]Sequential Feature Selection (SFS)\n')

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)

    model = XGBRegressor()
    k_fold = KFold(shuffle=True, random_state=123)

    # Selection on all features
    SFS = SequentialFeatureSelector(
        estimator=model,
        k_features=(1, 62),
        verbose=2,
        cv=k_fold,
        n_jobs=1,
    )

    with warnings.catch_warnings(action='ignore'):
        SFS.fit(X_train, y_train)

    if save:
        with open(save_path / 'sfs.joblib', 'wb') as file:
            joblib.dump(SFS, file)

        print('\n[green]---- Successfully saved ----\n')

    return SFS


if __name__ == '__main__':
    DATASET_PATH = Path('D:/dados_tcc/dataset')
    SAVE_PATH = Path('D:/dados_tcc/files')

    X = pd.read_feather(DATASET_PATH / 'X_clean.feather')
    y = np.load(DATASET_PATH / 'y_clean.npy')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    sfs = sequential_feature_selection(
        X_train, y_train, save=True, save_path=SAVE_PATH
    )

    with open(SAVE_PATH / 'sfs_metrics.joblib', 'wb') as file:
        joblib.dump(sfs.get_metric_dict(), file)
