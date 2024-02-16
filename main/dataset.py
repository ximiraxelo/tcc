import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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


def make_dataset(system, fs=10_000, window_size=10_000, tf=20, y0=[0, 0]):
    t = np.linspace(0, tf, fs * tf)
    u = np.ones(fs * tf, dtype=int)

    parameters = parameters_generator(
        phi0_range=np.linspace(0, 10, 150).tolist(),
        phif_range=np.linspace(0, 10, 150).tolist(),
        t0_range=np.linspace(0, 20, 50).tolist(),
    )

    for index, parameter in enumerate(parameters):

        # First state-space variable
        x1 = solve_ivp(
            system,
            t_span=[0, tf],
            y0=y0,
            dense_output=True,
            vectorized=True,
            args=parameter,
        ).sol(t)[0]

        # TODO: parameter

        # Sliding window
        signals = np.vstack((x1, u))
        windowed = sliding_window_view(signals, window_shape=(2, window_size))
        windowed = np.squeeze(windowed)
