from typing import Tuple, List
import numpy as np
from jaxtyping import Float, jaxtyped
from beartype import beartype

@jaxtyped(typechecker=beartype)
def approximate_intersect(points: List[Float[np.ndarray, "3 1"]], direction_vectors: List[Float[np.ndarray, "3 1"]]) -> Float[np.ndarray, "3 1"]:
    """
    Compute the point minimising the squared distance between n lines.

    Source: https://math.stackexchange.com/questions/36398/point-closest-to-a-set-four-of-lines-in-3d

    points: List of (3, 1) arrays
    direction_vectors: List of (3, 1) arrays
    """
    
    if len(points) != len(direction_vectors):
        raise ValueError("Need to provide exactly as many points p_i as direction vectors d_i")

    M = np.zeros((3, 3)) # M = Sum of all A_i
    b = np.zeros((3, 1)) # b = Sum of all A_i * p_i

    for i in range(len(points)):
        p_i = points[i]
        d_i = direction_vectors[i]
        d_i = d_i / np.linalg.norm(d_i)

        # A_i = I - d_i*d_i^T
        A_i = np.eye(3) - d_i @ d_i.T
        
        M += A_i
        b += A_i @ p_i
    
    computed_point_x = np.linalg.solve(M, b)

    return computed_point_x