import pickle
from pathlib import Path

import pandas as pd
from rich import print

DATA_PATH = Path('D:/dados_tcc/dados_tcc_esdras')
SAVE_PATH = Path('D:/dados_tcc/files')

if not SAVE_PATH.exists():
    SAVE_PATH.mkdir()

filtered_columns = []

for column in range(730):

    feature_list = []

    for dataset_path in DATA_PATH.glob('**/X.feather'):
        feature_chunk = pd.read_feather(
            dataset_path, columns=[column]
        ).squeeze()
        feature_list.append(feature_chunk)

    feature = pd.concat(feature_list, ignore_index=True)

    print(f'[{column:3}] {feature.name:35} | ', end='')

    # Non zero variance and not null features
    if feature.var() > 1e-6 and feature.notna().all():
        filtered_columns.append(column)
        print('[green]OK')
    else:
        print('[red]REMOVED')

with open(SAVE_PATH / 'non_zero_variance_columns.pickle', 'wb') as file:
    pickle.dump(filtered_columns, file)
