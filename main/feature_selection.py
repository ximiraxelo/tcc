import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from scipy.stats import pearsonr


def non_zero_variance_selector(
    data_path, save_path=None, save=False, threshold=0
):

    print('[purple]Non Zero Variance Selector\n')

    selected_features = []

    for column in range(730):

        feature_list = []

        for dataset_path in data_path.glob('**/X.feather'):
            feature_chunk = pd.read_feather(
                dataset_path, columns=[column]
            ).squeeze()
            feature_list.append(feature_chunk)

        feature = pd.concat(feature_list, ignore_index=True)

        print(f'[{column:3}] {feature.name:35} | ', end='')

        # Non zero variance and not null features
        if feature.var() > threshold and feature.notnull().all():
            selected_features.append(column)
            print('[green]OK')
        else:
            print('[red]REMOVED')

    print(f'\nRemaining features: {len(selected_features)}')

    if save:
        with open(
            save_path / 'non_zero_variance_features.pickle', 'wb'
        ) as file:
            pickle.dump(selected_features, file)

        print('\n[green]---- Successfully saved ----\n')

    return selected_features


def correlation_selector(
    previous_features, data_path, save_path=None, save=False, threshold=1e-3
):

    print('[purple]Correlation Selector\n')

    selected_features = []
    target_list = []

    for target_path in data_path.glob('**/y.npy'):
        target_chunk = np.load(target_path)
        target_list.append(target_chunk)

    target = np.concatenate(target_list)

    for column in previous_features:

        feature_list = []

        for dataset_path in data_path.glob('**/X.feather'):
            feature_chunk = pd.read_feather(
                dataset_path, columns=[column]
            ).squeeze()
            feature_list.append(feature_chunk)

        feature = pd.concat(feature_list, ignore_index=True)

        with warnings.catch_warnings(action='ignore'):
            correlation = pearsonr(feature, target).statistic

        print(f'[{column:3}] {feature.name:35} | ', end='')

        if abs(correlation) > threshold:
            selected_features.append(column)
            print(f'[green]{correlation:7.4f} [/]| [green]OK')
        else:
            print(f'[red]{correlation:7.4f} [/]| [red]REMOVED')

    print(f'\nRemaining features: {len(selected_features)}')

    if save:
        with open(save_path / 'correlation_features.pickle', 'wb') as file:
            pickle.dump(selected_features, file)

        print('\n[green]---- Successfully saved ----\n')

    return selected_features


if __name__ == '__main__':

    DATA_PATH = Path('D:/dados_tcc/dados_tcc_esdras')
    SAVE_PATH = Path('D:/dados_tcc/files')

    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir()

    non_zero_variance_features = non_zero_variance_selector(
        DATA_PATH, SAVE_PATH, save=True
    )

    correlation_features = correlation_selector(
        non_zero_variance_features,
        DATA_PATH,
        SAVE_PATH,
        save=True,
        threshold=0.2,
    )
