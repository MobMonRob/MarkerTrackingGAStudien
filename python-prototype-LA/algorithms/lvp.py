from typing import Tuple
import numpy as np
from jaxtyping import Float


""" def lotfusspunkt_verfahren_diy(g: line, h: line) -> (np.array, np.array, np.array):
    lambda_mu_matrix = np.array([[np.dot(-g.d, g.d), np.dot(h.d, g.d)],[np.dot(-g.d, h.d), np.dot(h.d, h.d)]])
    p_vector = np.array([[np.dot(h.p-g.p, g.d)], [np.dot(h.p-g.p, h.d)]])

    lambda_mu_vector = np.linalg.solve(lambda_mu_matrix, -p_vector) # minus nicht vergessen weil rübergezogen
    lambda_ = lambda_mu_vector[0]
    mu = lambda_mu_vector[1]

    orthogonal_distance_vector = - g.d * lambda_ + h.d * mu + (h.p - g.p)
    laufpunkt_g = g.p + lambda_ * g.d
    laufpunkt_h = h.p + mu * h.d
    return orthogonal_distance_vector, laufpunkt_g, laufpunkt_h """

def lotfusspunkt_verfahren(
    p_g: Float[np.ndarray, "3 1"],
    d_g: Float[np.ndarray, "3 1"],
    p_h: Float[np.ndarray, "3 1"],
    d_h: Float[np.ndarray, "3 1"],
) -> Tuple[Float[np.ndarray, "3 1"], Float[np.ndarray, "3 1"], Float[np.ndarray, "3 1"]]:
    """
    Compute the shortest connecting vector between two 3D lines

    p_g, d_g, p_h, d_h : (3, 1) arrays
    p_g and p_h are points on lines g and h; d_g and d_h are their direction vectors.
    """
    pg = np.asarray(p_g, dtype=float).reshape(-1)
    dg = np.asarray(d_g, dtype=float).reshape(-1)
    ph = np.asarray(p_h, dtype=float).reshape(-1)
    dh = np.asarray(d_h, dtype=float).reshape(-1)

    if not (pg.size == dg.size == ph.size == dh.size == 3):
        raise ValueError("All inputs must be 1D arrays with exactly 3 elements.")
    if np.allclose(dg, 0) or np.allclose(dh, 0):
        raise ValueError("Direction vectors d_g and d_h must be non-zero.")

    # Build 2x2 system for lambda and mu
    m11 = np.dot(-dg, dg)
    m12 =  np.dot(dh, dg)
    m21 = np.dot(-dg, dh)
    m22 =  np.dot(dh, dh)
    A = np.array([[m11, m12],
                  [m21, m22]], dtype=float)

    delta_p = ph - pg
    b = -np.array([np.dot(delta_p, dg), np.dot(delta_p, dh)], dtype=float)

    try:
        lambda_, mu = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        raise ValueError("Lines are parallel or numerically ill-conditioned (singular 2x2 system).") from e

    orthogonal_distance_vector = -dg * lambda_ + dh * mu + delta_p
    footpoint_g = pg + lambda_ * dg
    footpoint_h = ph + mu * dh
    return orthogonal_distance_vector.reshape(3, 1), footpoint_g.reshape(3, 1), footpoint_h.reshape(3, 1)
