import math
import numpy as np
from jaxtyping import Float
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from model.camera import Camera
from typing import List, Dict
from model.camera import Camera
from matplotlib.colors import Normalize
from geometry.rays import ray_batch_from_to

def create_3d_figure() -> tuple[plt.Figure, plt.Axes]:
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=(-1600, 2000), ylim=(-1500, 2000), zlim=(-2000, 2000))
    ax.set_aspect('equal', 'box')

    ax.set_title("Visualization of 3D Scene")

    return fig, ax

def plot_camera(ax: plt.Axes, camera: Camera, cameras_total: int = 13, scale_cam_coords_relative_to_focal: int = 3, cmap='tab20') -> None:
    
    principal_ccs = camera.image_to_ccs(camera.get_principal_point())
    principal_ccs_scaled = principal_ccs * camera.get_focal_length_x()/scale_cam_coords_relative_to_focal
    principal_wcs = camera.ccs_to_wcs(principal_ccs_scaled)

    norm = Normalize(vmin=0, vmax=cameras_total - 1)

    ax.scatter(*camera.get_position(), c=camera.user_id, cmap=cmap, s=30, marker='o', norm=norm)
    ax.scatter(*principal_wcs, c=camera.user_id, cmap=cmap, s=10, marker='o', norm=norm)

    draw_virtual_image_plane(ax, camera, cameras_total, scale_cam_coords_relative_to_focal=scale_cam_coords_relative_to_focal)

    norm = Normalize(vmin=0, vmax=cameras_total - 1)
    cmap = plt.cm.tab20
    color = cmap(norm(camera.user_id))

    for wcs_point in camera.get_all_wcs_points():
        ray = ray_batch_from_to(camera.get_position(), wcs_point)
        ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color=color, linestyle='dashed')

#TODO REWRITE!
def draw_virtual_image_plane(ax: plt.Axes, camera: Camera, cameras_total: int = 13, scale_cam_coords_relative_to_focal: int = 3):
    sensor_corners = []

    for i,j in [[0, 0], [0, 1088], [2048, 1088], [2048, 0]]:

        pt_ccs = camera.image_to_ccs(centroid = np.reshape(np.array([i,j]), (2,1)))
        pt_ccs_scaled = pt_ccs * camera.get_focal_length_x()/scale_cam_coords_relative_to_focal
        pt_wcs = camera.ccs_to_wcs(pt_ccs_scaled)

        sensor_corners.append(pt_wcs[:3].flatten().tolist())

    image_plane_verts = [sensor_corners]

    plane = Poly3DCollection(image_plane_verts, alpha=0.3, edgecolor='k')
    
    norm = Normalize(vmin=0, vmax=cameras_total - 1)
    cmap = plt.cm.tab20
    color = cmap(norm(camera.user_id))
    
    plane.set_facecolor(color)

    ax.add_collection3d(plane)

def scatter_3d_markers(ax: plt.Axes, points: List[Float[np.ndarray, "3 1"]], color="orange"):
    for point in points:
        ax.scatter(*point, color=color, s = 10, marker='x')