import time
import warnings
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import tsfel
from scipy.integrate import solve_ivp


def system(t, x, phi0, phif, t0):
    """_summary_

    Args:
        t (float):  simulation time in seconds
        x (list): _description_
        phi0 (float): lower limit of the LPV parameter
        phif (float): upper limit of the LPV parameter
        t0 (float): moment when the LPV parameter changes, in seconds
    """

    # LPV parameter
    phi = phi0 if t <= t0 else phif

    # Input signal
    u = 1

    # System definition in state-space
    A = np.array([[0, phi], [-phi, 0]])
    B = np.array([[0], [1]])

    dx = np.dot(A, x) + np.dot(B, u)

    return dx


def parameters_generator(phi0_range, phif_range, t0_range):
    for phi0 in phi0_range:
        for phif in phif_range:
            for t0 in t0_range:
                yield (phi0, phif, t0)


def backup(X, y, backup_path):
    try:
        X.to_feather(backup_path / f'X_backup.feather')
        np.save(backup_path / f'y_backup.npy', y)
    except Exception as err:
        print('\nxxxx Backup error xxxx\n')
        print(err)
    else:
        print('\n---- Backup performed successfully ----\n')


def make_dataset(
    system,
    fs=10_000,
    window_size=10_000,
    overlap=0.75,
    tf=20,
    y0=[0, 0],
    params_range=(20, 20, 50),
):

    N_SYSTEMS = np.prod(params_range)
    WINDOW_STEP = int(round(window_size * (1 - overlap)))
    N_WINDOWS = ceil((fs * tf - (window_size - 1)) / WINDOW_STEP)
    SLICES = slice(window_size - 1, fs * tf, WINDOW_STEP)

    t = np.linspace(0, tf, fs * tf)
    u = np.ones(fs * tf, dtype=int)

    parameters = parameters_generator(
        phi0_range=np.linspace(0.01, 10, params_range[0]).tolist(),
        phif_range=np.linspace(0.01, 10, params_range[1]).tolist(),
        t0_range=np.linspace(1, 19, params_range[2]).tolist(),
    )

    cfg = tsfel.get_features_by_domain(json_path='features.json')
    total_time = 0

    X = pd.DataFrame()
    phi = np.empty(N_SYSTEMS * N_WINDOWS)

    for index, parameter in enumerate(parameters):

        initial_time = time.time()
        print(f'System {index+1:6_} of {N_SYSTEMS:_}', end=' ')

        # First state-space variable
        x1 = solve_ivp(
            system,
            t_span=[0, tf],
            y0=y0,
            dense_output=True,
            vectorized=True,
            args=parameter,
        ).sol(t)[0]

        # LPV parameter
        phi0, phif, t0 = parameter
        phi_slices = slice(index * N_WINDOWS, (index + 1) * N_WINDOWS)
        phi[phi_slices] = np.where(t <= t0, phi0, phif)[SLICES]

        # Feature extraction
        X_dataframe = pd.DataFrame({'x1': x1, 'u': u})
        with warnings.catch_warnings(action='ignore'):
            features = tsfel.time_series_features_extractor(
                cfg,
                X_dataframe,
                fs=10_000,
                window_size=window_size,
                overlap=overlap,
                verbose=0,
            )
        X = pd.concat([X, features], ignore_index=True)

        final_time = time.time()
        loop_time = final_time - initial_time
        total_time += loop_time / 60
        print(f'| {loop_time:4.2f} s | {total_time:6.2f} m')

    return X, phi


if __name__ == '__main__':
    X, y = make_dataset(system, overlap=0.6, tf=10)

    SAVE_PATH = Path('D:') / 'dados_tcc' / 'step'

    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir()

    X.to_feather(SAVE_PATH / 'X.feather')
    np.save(SAVE_PATH / 'y.npy', y)
