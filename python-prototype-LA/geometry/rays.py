import numpy as np
from jaxtyping import Float

def ray_from_to(
    p_from: Float[np.ndarray, "3 1"],
    p_to: Float[np.ndarray, "3 1"],
    t: float,
) -> Float[np.ndarray, "3 1"]:
    origin = np.asarray(p_from, dtype=float)
    target = np.asarray(p_to, dtype=float)

    if origin.shape != (3, 1) or target.shape != (3, 1):
        raise ValueError(f"ray_from_to expects (3,1) column vectors; got {origin.shape} and {target.shape}")

    return origin + float(t) * (target - origin)

def ray_batch_from_to(
    p_from: Float[np.ndarray, "3 1"],
    p_to: Float[np.ndarray, "3 1"],
    t_values: Float[np.ndarray, "N"] = np.linspace(0.0, 3000.0, 10),
) -> Float[np.ndarray, "N 3"]:
    """
    Batched variant: compute points along the ray for a set of t values.
    Returns an (N, 3) array where each row is [x, y, z] for the corresponding t.
    """
    origin = np.asarray(p_from, dtype=float).reshape(3,)   # (3,)
    target = np.asarray(p_to, dtype=float).reshape(3,)     # (3,)
    t_vals = np.asarray(t_values, dtype=float).reshape(-1) # (N,)

    direction = target - origin                             # (3,)
    points = origin + t_vals[:, None] * direction           # (N, 3)
    return points