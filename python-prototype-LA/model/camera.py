from __future__ import annotations

import numpy as np
from jaxtyping import Float
from typing import List, Tuple, Dict
from iotools.camera_config_parser import CameraConfig
from scipy.spatial.transform import Rotation as R
import model.calibration as calibration
import model.distortion as distortion
import geometry.transforms as transforms

class Camera():
    def __init__(self, user_id: int, 
                 config_matrix: Float[np.ndarray, "3 3"],
                 extrinsic_matrix: Float[np.ndarray, "4 4"],
                 distortion_center: Float[np.ndarray, "2 1"],
                 k1: float,
                 k2: float,
                 k3: float,
                 sensor_xy: Float[np.ndarray, "2 1"],
                 centroids: List[Float[np.ndarray, "2 1"]],
                 correct_distortion: bool = True) -> None:
        
        self.user_id = user_id
        self.config_matrix = config_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.distortion_center=distortion_center
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.sensor_xy=sensor_xy
        self.centroids = centroids
        self.correct_distortion = correct_distortion
        self.correspondences: Dict[int, List[Tuple[int, int, float]]] = {}  # Maps centroid index to list of (other_camera_id, other_centroid_idx, epipolar_distance) tuples

        self.init_correspondences()

    def init_correspondences(self):
        # Initialize correspondences with empty lists for each centroid index
        for i in range(len(self.centroids)):
            self.correspondences[i] = []

    def get_centroids_map(self) -> Dict[int, Float[np.ndarray, "2 1"]]:
        """Return a mapping from centroid index to (optionally undistorted) centroid."""
        return {i: c for i, c in enumerate(self.get_centroids())}

    def get_all_wcs_points(self) -> List[Float[np.ndarray, "3 1"]]:
        wcs_points = []

        for centroid in self.get_centroids():

            point_ccs = self.image_to_ccs(centroid)
            point_wcs = self.ccs_to_wcs(point_ccs)

            wcs_points.append(transforms.dehomogenize(point_wcs))
        
        return wcs_points
    
    def image_to_ccs(self, centroid: Float[np.ndarray, "2 1"]) -> Float[np.ndarray, "3 1"]:
        centroid_homog = transforms.homogenize(centroid)
        return np.linalg.inv(self.config_matrix) @ centroid_homog
    
    def ccs_to_wcs(self, ccs_point: Float[np.ndarray, "3 1"]) -> Float[np.ndarray, "4 1"]:
        ccs_homog = transforms.homogenize(ccs_point)
        return np.linalg.inv(self.extrinsic_matrix) @ ccs_homog
    
    def get_centroids(self) -> List[Float[np.ndarray, "2 1"]]:
        if(self.correct_distortion):
            centroids_undistorted = []
            for centroid in self.centroids:
                centroids_undistorted.append(distortion.undistort(centroid, self.distortion_center, self.get_principal_point() ,self.k1, self.k2, self.k3))
            return centroids_undistorted
        else:
            return self.centroids

    def get_focal_length_x(self) -> float:
        return self.config_matrix[0, 0]

    def get_focal_length_y(self) -> float:
        return self.config_matrix[1, 1]

    def get_principal_point(self) -> Float[np.ndarray, "2 1"]:
        principal_x = self.config_matrix[0, 2]
        principal_y = self.config_matrix[1, 2]
        return np.array([[principal_x],[principal_y]])

    def get_position(self) -> Float[np.ndarray, "3 1"]:
        translation = self.extrinsic_matrix[:3, 3].reshape(3,1)
        rotation = self.extrinsic_matrix[:3, :3]
        return - rotation.T @ translation # t = -R * c_w => c_w = -R^T * t

    def get_rotation_matrix(self) -> Float[np.ndarray, "3 3"]:
        return self.extrinsic_matrix[:3, :3]

    def project(self, point_wcs: Float[np.ndarray, "3 1"]) -> Float[np.ndarray, "2 1"]:
        """Project a 3D world-coordinate point onto the image plane, returning a 2D pixel coordinate."""
        point_ccs = self.extrinsic_matrix @ transforms.homogenize(point_wcs)
        point_img_h = self.config_matrix @ point_ccs[:3]
        return transforms.dehomogenize(point_img_h)
    
    def compute_epipolar_line(self, other_camera: Camera, other_centroid: Float[np.ndarray, "2 1"]) -> Tuple[Float[np.ndarray, "2 1"], Float[np.ndarray, "2 1"]]:
        """
        Compute the epipolar line parameters p and d for the epipolar line p + lambda * d in this camera's image plane.
        
        Takes the other camera and a centroid as observed by the other camera.
        """
        epipole = self.project(other_camera.get_position())
        point_wcs = other_camera.ccs_to_wcs(other_camera.image_to_ccs(other_centroid))
        point_pixel = self.project(transforms.dehomogenize(point_wcs))
        direction = point_pixel - epipole

        return epipole, direction

    def closest_centroid(self, epipole: Float[np.ndarray, "2 1"], direction: Float[np.ndarray, "2 1"]) -> Tuple[int, float]:
        """
        Find the centroid closest to the epipolar line defined by epipole and direction. 

        Returns the index of the closest centroid and its distance to the epipolar line.
        """
        min_distance = float('inf')
        closest_idx = None
        d = direction.flatten()

        for i, centroid in enumerate(self.get_centroids()):
            v = centroid.flatten() - epipole.flatten()
            _lambda = np.dot(v, d) / np.dot(d, d)
            distance = np.linalg.norm(epipole.flatten() + _lambda * d - centroid.flatten())
            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        return closest_idx, min_distance

    def compute_correspondences_with(self, other_camera: Camera, max_dist: float = 1.) -> None:
        """
        Compute correspondences between this camera and another camera by finding the closest centroid to the epipolar line for each point observed by the other camera.

        Stores results in self.correspondences as a mapping from this camera's centroids to lists of (other_camera_id, other_centroid) pairs.
        """
        for other_idx, other_centroid in other_camera.get_centroids_map().items():
            epipole, direction = self.compute_epipolar_line(other_camera, other_centroid)
            closest_idx, distance = self.closest_centroid(epipole, direction)

            # #enforcing bidirectionality
            # if closest_idx is not None:  
            #     epipole, direction = other_camera.compute_epipolar_line(self, self.get_centroids_map()[closest_idx])  
            #     closest_index_other, distance_other = other_camera.closest_centroid(epipole, direction)
            # else: 
            #     continue

            #if closest_index_other == other_idx and distance <= max_dist and distance_other <= max_dist:
            if closest_idx is not None and distance <= max_dist: 
                self.correspondences[closest_idx].append((other_camera.user_id, other_idx, distance))

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
    extrinsic_matrix = calibration.build_extrinsic_matrix(rotation_matrix, config.position)

    return Camera(
        user_id=config.user_id,
        config_matrix=config_matrix,
        extrinsic_matrix=extrinsic_matrix,
        distortion_center=config.distortion_center,
        k1=config.k1,
        k2=config.k2,
        k3=config.k3,
        sensor_xy=config.sensor_xy,
        centroids=centroids,
    )