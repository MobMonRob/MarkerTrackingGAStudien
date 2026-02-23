from __future__ import annotations
from jaxtyping import Float
import xml.etree.ElementTree as ET
from typing import List
import numpy as np
from dataclasses import dataclass

@dataclass
class CameraConfig:
    """
    Parsed camera configuration for one camera, based on the prototype logic.
    """
    user_id: int
    principal_point: Float[np.ndarray, "2 1"]       # [[px], [py]]
    focal_length: float                             # scalar; used for both focal_x and focal_y
    position: Float[np.ndarray, "3 1"]              # [[x], [y], [z]]
    orientation_xyzw: Float[np.ndarray, "4"]        # quaternion [x, y, z, w]
    distortion_center: Float[np.ndarray, "2 1"]     # [[x], [y]]
    k1: float                                       # 
    k2: float                                       # distortion parameters
    k3: float                                       #
    sensor_xy: Float[np.ndarray, "2 1"]

def parse_file(xml_path: str) -> List[CameraConfig]:
    """
    Parse an XCP XML file and return a list of CameraConfig objects for each <Camera>.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return parse_root(root)


def parse_root(root: ET.Element) -> List[CameraConfig]:
    """
    Parse from a pre-obtained XML root element.
    """
    configs: List[CameraConfig] = []
    for cam in root.iter("Camera"):
        configs.append(_parse_camera_element(cam))
    return configs

def _parse_camera_element(cam_el: ET.Element) -> CameraConfig:
    """
    Parse a single <Camera> element.
    Uses the first <KeyFrame> child for attributes.
    """
    user_id_str = cam_el.get("USERID")
    user_id = int(user_id_str)

    sensor_size_str= cam_el.get("SENSOR_SIZE")
    sensor_xy = np.fromstring(sensor_size_str, sep=" ")
    sensor_xy = sensor_xy.astype(float)

    key_frames = list(cam_el.iter("KeyFrame"))
    key_frame = key_frames[0]

    principal_point_str = key_frame.get("PRINCIPAL_POINT")
    focal_length_str = key_frame.get("FOCAL_LENGTH")
    position_str = key_frame.get("POSITION")
    orientation_str = key_frame.get("ORIENTATION")
    vicon_radial2_str = key_frame.get("VICON_RADIAL2")

    principal_point = np.fromstring(principal_point_str, sep=" ")
    principal_point = principal_point.reshape(2, 1)
    principal_point = principal_point.astype(float)

    focal_length = float(focal_length_str)

    position = np.fromstring(position_str, sep=" ")
    position = position.astype(float)

    orientation_xyzw = np.fromstring(orientation_str, sep=" ")
    orientation_xyzw = orientation_xyzw.astype(float)

    distortion = _parse_vicon_radial2(vicon_radial2_str)
    distortion_center_x = float(distortion[0])
    distortion_center_y = float(distortion[1])
    distortion_center = np.array([[distortion_center_x],[distortion_center_y]]).astype(float)
    k1 = float(distortion[2])
    k2 = float(distortion[3])
    k3 = float(distortion[4])

    return CameraConfig(
        user_id=user_id,
        principal_point=principal_point,           # Float[np.ndarray, "2 1"]
        focal_length=focal_length,                 # float
        position=position,                         # Float[np.ndarray, "3 1"]
        orientation_xyzw=orientation_xyzw,         # Float[np.ndarray, "4"]
        distortion_center=distortion_center,       # Float[np.ndarray, "2 1"]
        k1=k1,
        k2=k2,
        k3=k3,
        sensor_xy=sensor_xy
    )

def _parse_vicon_radial2(value: str) -> np.ndarray:
    """
    Parse the VICON_RADIAL2 string, removing the 'Vicon3Parameter '.
    Returns an array of floats: [cx, cy, k1, k2, k3].
    """
    prefix = "Vicon3Parameter "
    if value.startswith(prefix):
        value = value[len(prefix):]
    arr = np.fromstring(value, sep=" ")

    return arr