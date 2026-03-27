"""
Full Code Reproducing the results in Chapter 3.3
"""

from iotools import camera_config_parser, csv_parser
from model.camera import *
from algorithms.lsp import approximate_intersect
import numpy as np
from visualisation.image_plane_grid import plot_image_plane_grid
import visualisation.scene as scene
import matplotlib.pyplot as plt

# Hard-coded paths
XCP_PATH = "../camera-configs/experiment-3.xcp"
CENTROIDS_PATH = "../csvdata/experiment-3/all-cams-masked/centroid-dump-1-marker-all-cams.csv"
MARKERS_PATH = "../csvdata/experiment-3/all-cams-masked/marker-dump-1-marker-all-cams.csv"

# Load data
configs = camera_config_parser.parse_file(XCP_PATH)
centroids_raw = csv_parser.load_centroid_data_raw(CENTROIDS_PATH)
markers = csv_parser.load_known_markers(MARKERS_PATH)

# Instantiate cameras (keyed by user_id string)
cams = {}
for config in configs:
    cam = from_params(config, csv_parser.centroids_for_camera_columns(centroids_raw, config.user_id))
    cams[cam.user_id] = cam

print(f"Actual:\n{markers[0]}")

for correct_distortion in (False, True):
    all_origins = []
    all_directions = []
    for cam in cams.values():
        cam.correct_distortion = correct_distortion

        if cam.get_all_wcs_points() != []:
            all_origins.append(cam.get_position())
            all_directions.append(cam.get_position() - cam.get_all_wcs_points()[0])

    computed = approximate_intersect(all_origins, all_directions)
    error = np.linalg.norm(computed - markers[0])

    print(f"Computed(correct distortion:{correct_distortion}):\n{computed}\nError={error}")

plot_image_plane_grid(cams.values())

# visualisation of 3D scene, getting only those cameras such that 4 lines intersect at the point
fig, ax = scene.create_3d_figure(cameras=cams.values())
for camera in cams.values():
    scene.plot_camera(ax, camera)

# Render & print the dumped points "Musterlösung"
scene.scatter_3d_markers(ax, markers, color="green")

fig.savefig("figure.svg", format="svg", dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)

plt.show()
