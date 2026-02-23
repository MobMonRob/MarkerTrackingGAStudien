import numpy as np
from jaxtyping import Float
from model import distortion

def build_config_matrix(focal_length_x : float, focal_length_y: float, principal_point: Float[np.ndarray, "2 1"]) -> Float[np.ndarray, "3 3"]:
    """
    Uses principal point and focal length to build the configuration matrix according to the linear camera model. 

    Returns a (3,3) Matrix representing camera intrinsics.
    """

    return np.array([[focal_length_x, 0, principal_point[0, 0]],
                    [0, focal_length_y, principal_point[1, 0]],
                    [0, 0, 1]])

def build_config_matrix_corrected(focal_length_x : float, focal_length_y: float, principal_point: Float[np.ndarray, "2 1"], distortion_center: Float[np.ndarray, "2 1"], k1: float, k2: float, k3: float) -> Float[np.ndarray, "3 3"]:
    """
    Uses principal point and focal length to build the configuration matrix according to the linear camera model. 
    Corrects the camera principal point for distortion.

    INFO: In the vicon system, principal_point == distortion_center. 
    -> since vicon provides only k1-k3, dones't account for tangential distortion!

    Returns a (3,3) Matrix representing camera intrinsics.
    """
    assert(np.array_equal(principal_point, distortion_center)) # see above
    
    # pointless calculation :)
    principal_point_undist = distortion.undistort(principal_point, distortion_center ,k1, k2, k3)
    assert(np.array_equal(principal_point, principal_point_undist)) # here we go again

    #TODO: EXPERIMENT: Remove translation to pixel coordinate system (principal point) and undistort beforehand?

    return np.array([[focal_length_x, 0, principal_point_undist[0, 0]],
                    [0, focal_length_y, principal_point_undist[1, 0]],
                    [0, 0, 1]])

def build_extrinsic_matrix(rotation_matrix: Float[np.ndarray, "3 3"], position: Float[np.ndarray, "3 1"]) -> Float[np.ndarray, "3 4"]:
    """
    Multiplying with this matrix performs rotation and translation of points from the WCS to the CCS in one operation. 

    Assumes that rotation matrix given performs world-to-camera rotation.

    c_w = position (Camera Pinhole in World Coordinates)
    x_c = R * (x_w - c_w) = R * x_w - R * c_w
    t = -R * c_w

    -> [x_c] = [R | t] @ [x_w; 1]

    Returns a (4,4) Matrix representing camera extrinsics.
    """
    translation = -rotation_matrix @ position   # t = -R * c_w
    extrinsic_matrix = np.eye(4, dtype=float)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation       # perform R | t
    return extrinsic_matrix
