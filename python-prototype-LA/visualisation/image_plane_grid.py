import math
import numpy as np
import matplotlib.pyplot as plt
from model.camera import Camera
from typing import List, Dict
from matplotlib.colors import Normalize

def plot_image_plane_grid(cameras: List[Camera]) -> Dict[int, plt.Axes]:
    """
    Create a grid of image planes and return a mapping
    from camera.user_id -> axes object.
    """
    square = math.ceil(math.sqrt(len(cameras)))

    fig = plt.figure()
    fig.subplots_adjust(left=0.01, bottom=None, right=0.99, top=0.95, wspace=0, hspace=1)

    axes_map: Dict[int, plt.Axes] = {}

    for camera in cameras:
        # Use a consistent ordering of subplots; here we assume user_id starts at 0
        ax = fig.add_subplot(square, square, camera.user_id + 1)

        ax.set_title(f"UserID {camera.user_id}")

        sensor_x, sensor_y = camera.sensor_xy.ravel()
        ax.set(xlim=(0, 200000), ylim=(0, 200000))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()

        distortionBool = camera.correct_distortion

        camera.correct_distortion = False
        for coordinate in camera.get_centroids():
            ax.scatter(*coordinate, color="#FF0000", s=60, marker='+')

        camera.correct_distortion = True
        for coordinate in camera.get_centroids():
            ax.scatter(*coordinate, color="#00FF00", s=60, marker='+')

        camera.correct_distortion = distortionBool

        axes_map[camera.user_id] = ax

    return axes_map

def plot_single_image_plane_figure(camera: Camera, show_distorted: bool = False, show_indices: bool = True) -> None:

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(xlim=(0, camera.sensor_xy[0]), ylim=(0, camera.sensor_xy[1]))
    ax.set_title(f"UserID {camera.user_id} image plane")
    plt.gca().invert_yaxis()

    distortionBool = camera.correct_distortion

    if show_distorted:
        camera.correct_distortion=False
        for coordinate in camera.get_centroids():
            ax.scatter(*coordinate, color="#FF0000", s=60, marker='+', label="distorted")
    
    camera.correct_distortion=True
    for i, coordinate in enumerate(camera.get_centroids()):
        ax.scatter(*coordinate, color="#00AE00", s=100, marker='+', label="distortion corrected")
        if show_indices:
            xy = coordinate.flatten()
            ax.annotate(str(i), (xy[0], xy[1]), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color="#00AE00")
    
    camera.correct_distortion = distortionBool

    return fig, ax

def plot_epipolar_line(camera: Camera, other: Camera, other_centroid, ax: plt.Axes, lambdas = np.linspace(-3000, 3000, 300), cameras_total: int = 13, cmap=plt.cm.tab20) -> None:
    epipole, direction = camera.compute_epipolar_line(other, other_centroid)
    epipole = epipole.flatten()
    direction = direction.flatten()

    norm = Normalize(vmin=0, vmax=cameras_total - 1)
    color = cmap(norm(other.user_id))

    points = np.array([epipole + l * direction for l in lambdas])
    ax.plot(points[:, 0], points[:, 1], color=color, label=f"Cam {other.user_id}")