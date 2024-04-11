from pathlib import Path

import numpy as np
import pandas as pd
from rich import print


def clean_duplicates(X, y):
    complete_dataset = pd.concat([X, pd.DataFrame(y, columns=['y'])], axis=1)

    unique_samples = complete_dataset.drop_duplicates(ignore_index=True)

    y_unique = unique_samples['y'].to_numpy()
    X_unique = unique_samples.drop(columns='y')

    return X_unique, y_unique


if __name__ == '__main__':

    DATA_PATH = Path('D:/dados_tcc/dataset')

    X = pd.read_feather(DATA_PATH / 'X.feather')
    y = np.load(DATA_PATH / 'y.npy')

    X_unique, y_unique = clean_duplicates(X, y)

    try:
        X_unique.to_feather(DATA_PATH / 'X_clean.feather')
        np.save(DATA_PATH / 'y_clean.npy', y_unique)
    except Exception as err:
        print('\n[red]xxxx Saving Error xxxx\n')
        print(err)
    else:
        print('\n[green]---- Saved Successfully ----\n')
