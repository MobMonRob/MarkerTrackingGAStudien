import numpy as np
from jaxtyping import Float

def homogenize(vec: Float[np.ndarray, "N 1"], w: float = 1.0) -> np.ndarray:
    """
    Generic homogenization: append a homogeneous coordinate w to an N×1 column vector.
    Convert a N×1 Euclidean column vector to a N+1×1 homogeneous column vector
    by appending a 1 as the last coordinate: [[x], [y], [...]] -> [[x], [y], [...], [1]].
    """
    
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(f"Expected a column vector of shape (N,1), got {arr.shape}")
    return np.vstack([arr, np.array([[float(w)]], dtype=float)])


def dehomogenize(vec_h: Float[np.ndarray, "N 1"]) -> Float[np.ndarray, "N-1 1"]:
    """
    Dehomogenize an N×1 homogeneous column vector to an (N-1)×1 Euclidean column vector
    by dividing the first N-1 coordinates by w (the last coordinate).
    """
    arr = np.asarray(vec_h, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 1 or arr.shape[0] < 2:
        raise ValueError(f"Expected an (N,1) column vector with N>=2, got {arr.shape}")
    w = arr[-1, 0]
    if np.isclose(w, 0.0):
        raise ValueError("Homogeneous coordinate w is zero or near zero; cannot dehomogenize.")
    return arr[:-1, :] / w
