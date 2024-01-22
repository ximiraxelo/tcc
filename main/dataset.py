import numpy as np


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

    # TODO: Input signal

    # System definition in state-space
    A = np.array([[0, phi], [-phi, 0]])
    B = np.array([[0], [1]])

    dx = np.dot(A, x) + np.dot(B, u)

    return dx
