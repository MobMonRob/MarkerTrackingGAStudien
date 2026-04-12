import math
import numpy as np
from jaxtyping import Float
from typing import List, Iterable, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from model.camera import Camera
from typing import List, Dict
from model.camera import Camera
from matplotlib.colors import Normalize
from geometry.rays import ray_batch_from_to

def compute_xy_plot_limits_from_cameras(
    cameras: Optional[Iterable[object]] = None,
    margin: float = 0.05,
    default_xlim: Tuple[float, float] = (-1600.0, 2900.0),
    default_ylim: Tuple[float, float] = (-1500.0, 2000.0)
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute x and y limits from an iterable of Camera-like objects.
    If cameras is None or empty, returns provided defaults.
    Each camera must implement get_position() returning an array-like with at least 2 components.
    """
    if cameras is None:
        return default_xlim, default_ylim

    xs = []
    ys = []
    for cam in cameras:
        pos = np.asarray(cam.get_position()).ravel()
        if pos.size < 2:
            raise ValueError("camera position must contain at least 2 components")
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))

    if len(xs) == 0:
        return default_xlim, default_ylim

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))

    def _apply_margin(a_min, a_max):
        rng = a_max - a_min
        pad = rng * margin if rng != 0 else 1.0
        return a_min - pad, a_max + pad

    return _apply_margin(xmin, xmax), _apply_margin(ymin, ymax)


def create_3d_figure(
    cameras: Optional[Iterable[object]] = None,
    view_coordinates: bool = False,
    margin: float = 0.05,
    zlim: Optional[Tuple[float, float]] = None,
    default_xlim: Tuple[float, float] = (-1600.0, 2900.0),
    default_ylim: Tuple[float, float] = (-1500.0, 2000.0)
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a 3D matplotlib figure. If cameras provided, compute x/y limits from their positions;
    otherwise use defaults.
    """
    xlim, ylim = compute_xy_plot_limits_from_cameras(
        cameras=cameras, margin=margin, default_xlim=default_xlim, default_ylim=default_ylim
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim if zlim is not None else (0, 2000),
    )
    ax.set_aspect('equal')

    if not view_coordinates:
        disable_coordinates(ax)

    return fig, ax

def disable_coordinates(ax: plt.Axes):
    # disable grid and ax numbering
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # hide every pane except floor
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(True)

    # make axis transparent
    ax.xaxis.line.set_color((0,0,0,0))
    ax.yaxis.line.set_color((0,0,0,0))
    ax.zaxis.line.set_color((0,0,0,0))

def plot_camera(ax: plt.Axes, camera: Camera, cameras_total: int = 13, scale_cam_coords_relative_to_focal: int = 3, cmap=plt.cm.tab20) -> None:
    
    principal_ccs = camera.image_to_ccs(camera.get_principal_point())
    principal_ccs_scaled = principal_ccs * camera.get_focal_length_x()/scale_cam_coords_relative_to_focal
    principal_wcs = camera.ccs_to_wcs(principal_ccs_scaled)

    norm = Normalize(vmin=0, vmax=cameras_total - 1)

    ax.scatter(*camera.get_position(), c=camera.user_id, cmap=cmap, s=30, marker='o', norm=norm, label=camera.user_id)
    ax.text(*camera.get_position().flatten(), f"  {camera.user_id}", fontsize=6, color='grey')
    ax.scatter(*principal_wcs, c=camera.user_id, cmap=cmap, s=10, marker='o', norm=norm)

    draw_virtual_image_plane(ax, camera, cameras_total, scale_cam_coords_relative_to_focal=scale_cam_coords_relative_to_focal, cmap=cmap)

    norm = Normalize(vmin=0, vmax=cameras_total - 1)
    color = cmap(norm(camera.user_id))

    for wcs_point in camera.get_all_wcs_points():
        ray = ray_batch_from_to(camera.get_position(), wcs_point)
        ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color=color, linestyle='dashed')

#TODO REWRITE!
def draw_virtual_image_plane(ax: plt.Axes, camera: Camera, cameras_total: int = 13, scale_cam_coords_relative_to_focal: int = 3, cmap = plt.cm.tab20):
    sensor_corners = []

    for i,j in [[0, 0], [0, 1088], [2048, 1088], [2048, 0]]:

        pt_ccs = camera.image_to_ccs(centroid = np.array([[i], [j]], dtype=float))
        pt_ccs_scaled = pt_ccs * camera.get_focal_length_x()/scale_cam_coords_relative_to_focal
        pt_wcs = camera.ccs_to_wcs(pt_ccs_scaled)

        sensor_corners.append(pt_wcs[:3].flatten().tolist())

    image_plane_verts = [sensor_corners]

    plane = Poly3DCollection(image_plane_verts, alpha=0.3, edgecolor='k')
    
    norm = Normalize(vmin=0, vmax=cameras_total - 1)
    color = cmap(norm(camera.user_id))
    
    plane.set_facecolor(color)

    ax.add_collection3d(plane)

def scatter_3d_markers(ax: plt.Axes, points: List[Float[np.ndarray, "3 1"]], size=10, color="orange", symbol='x'):
    for point in points:
        ax.scatter(*point, color=color, s = size, marker=symbol)


def plot_camera_grey(ax: plt.Axes, camera: Camera, scale_cam_coords_relative_to_focal: int = 3) -> None:
    """Plot a camera in grey (position dot + virtual image plane), no rays."""
    ax.scatter(*camera.get_position(), c='grey', s=30, marker='o')
    ax.text(*camera.get_position().flatten(), f"  {camera.user_id}", fontsize=6, color='grey')

    sensor_corners = []
    for i, j in [[0, 0], [0, 1088], [2048, 1088], [2048, 0]]:
        pt_ccs = camera.image_to_ccs(centroid=np.array([[i], [j]], dtype=float))
        pt_ccs_scaled = pt_ccs * camera.get_focal_length_x() / scale_cam_coords_relative_to_focal
        pt_wcs = camera.ccs_to_wcs(pt_ccs_scaled)
        sensor_corners.append(pt_wcs[:3].flatten().tolist())

    plane = Poly3DCollection([sensor_corners], alpha=0.15, edgecolor='grey')
    plane.set_facecolor('lightgrey')
    ax.add_collection3d(plane)


def plot_cluster_rays(ax: plt.Axes, cams: Dict, components: List, marker_positions: List,
                      cmap=plt.cm.tab20) -> None:
    """
    For each cluster, draw the rays from each contributing camera in the cluster's color,
    and scatter the triangulated marker position.
    """
    import geometry.transforms as transforms
    from geometry.rays import ray_batch_from_to

    for i, (component, marker_pos) in enumerate(zip(components, marker_positions)):
        color = cmap(i % 20)

        # Plot triangulated marker
        if marker_pos is not None:
            ax.scatter(*marker_pos.flatten(), color=color, s=80, marker='x', zorder=10)

        # Plot rays from each camera in this cluster
        for (cam_id, centroid_idx) in component:
            cam = cams[str(cam_id)]
            centroid = cam.get_centroids_map()[centroid_idx]
            point_ccs = cam.image_to_ccs(centroid)
            point_wcs = transforms.dehomogenize(cam.ccs_to_wcs(point_ccs))

            ray = ray_batch_from_to(cam.get_position(), point_wcs, np.linspace(0, 3500, 10))
            ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color=color, linestyle='dashed', linewidth=0.8)