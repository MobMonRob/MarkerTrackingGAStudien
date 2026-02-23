import numpy as np
from typing import List
from jaxtyping import Float

def load_centroid_data_raw(csv_path: str, delimiter: str = ",") -> np.ndarray:
    """
    Load the centroids CSV into a structured NumPy array (names=True).

    Fields: cameraUserId, positionX, positionY.
    """
    arr = np.genfromtxt(csv_path, delimiter=delimiter, names=True)
    return arr

def centroids_for_camera_columns(
    centroids_raw: np.ndarray, user_id: int
) -> List[Float[np.ndarray, "2 1"]]:
    """
    Filter centroids by cameraUserId and return a list of (2,1) column vectors [[x],[y]].
    """
    cam_ids = centroids_raw["cameraUserId"]

    mask = cam_ids.astype(int) == int(user_id)
    rows = centroids_raw[mask]
    if rows.size == 0:
        return []

    xs = rows["positionX"].astype(float)    # x values of all centroids detected
    ys = rows["positionY"].astype(float)    # y values for all centroids detected
    return [_to_column_vector([x, y], 2) for x, y in zip(xs, ys)]

def _load_marker_data_raw(csv_path: str, delimiter: str = ",") -> np.ndarray:
    """
    Load marker/master data into a structured NumPy array (names=True).
    """
    arr = np.genfromtxt(csv_path, delimiter=delimiter, names=True)
    return arr

def load_known_markers(csv_path: str, delimiter: str = ",") -> List[Float[np.ndarray, "3 1"]]:
    """
    Load marker/master data and return a list of (3,1) column vectors [[x],[y],[z]].
    Handles single-row and multi-row CSVs.
    """
    arr = _load_marker_data_raw(csv_path, delimiter=delimiter)

    x = arr["markerPosX"].astype(float)     # x values of all markers detected
    y = arr["markerPosY"].astype(float)     # y values of all markers detected
    z = arr["markerPosZ"].astype(float)     # z values of all markers detected

    # Single-row case
    if np.ndim(x) == 0:
        return [_to_column_vector([float(x), float(y), float(z)], 3)]

    return [_to_column_vector([float(ix), float(iy), float(iz)], 3) for ix, iy, iz in zip(x, y, z)]

def _to_column_vector(vec: np.ndarray | List[float], dim: int) -> Float[np.ndarray, "dim 1"]:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} elements, got {arr.size}.")
    return arr.reshape(dim, 1)