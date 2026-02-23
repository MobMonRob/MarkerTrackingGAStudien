import numpy as np
from jaxtyping import Float
from typing import List
from iotools.camera_config_parser import CameraConfig
from scipy.spatial.transform import Rotation as R
import model.calibration as calibration
import model.distortion as distortion
import geometry.transforms as transforms

class Camera():
    def __init__(self, user_id: int, 
                 config_matrix: Float[np.ndarray, "3 3"],
                 config_matrix_corrected: Float[np.ndarray, "3 3"],
                 extrinsic_matrix: Float[np.ndarray, "3 4"],
                 distortion_center: Float[np.ndarray, "2 1"],
                 k1: float,
                 k2: float,
                 k3: float,
                 sensor_xy: Float[np.ndarray, "2 1"],
                 centroids: List[Float[np.ndarray, "2 1"]],
                 correct_distortion: bool = True) -> None:
        
        self.user_id = user_id
        self.config_matrix = config_matrix
        self.config_matrix_corrected = config_matrix_corrected
        self.extrinsic_matrix = extrinsic_matrix
        self.distortion_center=distortion_center
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.sensor_xy=sensor_xy
        self.centroids = centroids
        self.correct_distortion = correct_distortion

    def get_all_wcs_points(self) -> List[Float[np.ndarray, "3 1"]]:
        wcs_points = []

        for centroid in self.get_centroids():

            point_ccs = self.image_to_ccs(centroid)
            point_wcs = self.ccs_to_wcs(point_ccs)

            wcs_points.append(transforms.dehomogenize(point_wcs))
        
        return wcs_points
    
    def image_to_ccs(self, centroid: Float[np.ndarray, "2 1"]) -> Float[np.ndarray, "3 1"]:
        centroid_homog = transforms.homogenize(centroid)
        if(self.correct_distortion):
            return np.linalg.inv(self.config_matrix_corrected) @ centroid_homog
        else:
            return np.linalg.inv(self.config_matrix) @ centroid_homog
    
    def ccs_to_wcs(self, ccs_point: Float[np.ndarray, "3 1"]) -> Float[np.ndarray, "3 1"]:
        ccs_homog = transforms.homogenize(ccs_point)
        return np.linalg.inv(self.extrinsic_matrix) @ ccs_homog
    
    def get_centroids(self) -> List[Float[np.ndarray, "3 1"]]:
        if(self.correct_distortion):
            centroids_undistorted = []
            for centroid in self.centroids:
                centroids_undistorted.append(distortion.undistort(centroid, self.distortion_center ,self.k1, self.k2, self.k3))
            return centroids_undistorted
        else:
            return self.centroids

    def get_focal_length_x(self) -> float:
        return self.config_matrix[0, 0]

    def get_focal_length_y(self) -> float:
        return self.config_matrix[1, 1]

    def get_principal_point(self) -> Float[np.ndarray, "2 1"]:
        if self.correct_distortion:
            principal_x = self.config_matrix_corrected[0, 2]
            principal_y = self.config_matrix_corrected[1, 2]
        else:
            principal_x = self.config_matrix[0, 2]
            principal_y = self.config_matrix[1, 2]
        return np.array([[principal_x],[principal_y]])

    def get_position(self) -> Float[np.ndarray, "3 1"]:
        translation = self.extrinsic_matrix[:3, 3].reshape(3,1)
        rotation = self.extrinsic_matrix[:3, :3]
        return - rotation.T @ translation # t = -R * c_w => c_w = -R^T * t

    def get_rotation_matrix(self) -> Float[np.ndarray, "3 3"]:
        return self.extrinsic_matrix[:3, :3]

def from_params(config: CameraConfig, centroids: List[Float[np.ndarray, "2 1"]]) -> Camera:
    """
    Build a Camera from a parsed CameraConfig and a list of (2,1) centroid column vectors.
    - principal_point: reshaped to (2,1)
    - position: reshaped to (3,1)
    - rotation_matrix: built from quaternion [x, y, z, w]
    - focal lengths: use config.focal_length for both axes (or fall back to focal_length_x/y if present)
    - k1/k2/k3: copied from config
    """

    # Rotation from quaternion [x, y, z, w]
    quat = np.asarray(config.orientation_xyzw, dtype=float).reshape(4,)
    rotation_matrix = R.from_quat(quat).as_matrix().astype(float)

    # Focal lengths: single value used for both axes by default
    focal_length_x = float(config.focal_length)
    focal_length_y = float(config.focal_length)
    
    config_matrix = calibration.build_config_matrix(focal_length_x, focal_length_y, config.principal_point)
    config_matrix_corrected = calibration.build_config_matrix_corrected(focal_length_x, focal_length_y, config.principal_point, config.distortion_center, config.k1, config.k2, config.k3)
    extrinsic_matrix = calibration.build_extrinsic_matrix(rotation_matrix, config.position)

    return Camera(
        user_id=config.user_id,
        config_matrix=config_matrix,
        config_matrix_corrected=config_matrix_corrected,
        extrinsic_matrix=extrinsic_matrix,
        distortion_center=config.distortion_center,
        k1=config.k1,
        k2=config.k2,
        k3=config.k3,
        sensor_xy=config.sensor_xy,
        centroids=centroids,
    )