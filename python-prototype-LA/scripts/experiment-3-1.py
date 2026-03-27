"""
Full Code Reproducing the results in Chapter 3.1
"""

from iotools import camera_config_parser, csv_parser
from model.camera import *
from algorithms.lvp import lotfusspunkt_verfahren
import numpy as np
from visualisation.image_plane_grid import plot_single_image_plane_figure
import matplotlib.pyplot as plt

# Hard-coded paths
XCP_PATH = "../camera-configs/experiment-3.xcp"
CENTROIDS_PATH = "../csvdata/experiment-3/4-and-7-masked/centroid-dump-1-marker-cam-4-and-7.csv"
MARKERS_PATH = "../csvdata/experiment-3/4-and-7-masked/marker-dump-1-marker-cam-4-and-7.csv"

# Load data
configs = camera_config_parser.parse_file(XCP_PATH)
centroids_raw = csv_parser.load_centroid_data_raw(CENTROIDS_PATH)
markers = csv_parser.load_known_markers(MARKERS_PATH)

# Instantiate cameras (keyed by user_id string)
cams = {}
for config in configs:
    cam = from_params(config, csv_parser.centroids_for_camera_columns(centroids_raw, config.user_id))
    cams[cam.user_id] = cam

# Helper to compute marker from a camera pair
def compute_pair(cam_a, cam_b):
    vector_gh, fp_g, fp_h = lotfusspunkt_verfahren(
        cam_a.get_position(),
        cam_a.get_position() - cam_a.get_all_wcs_points()[0],
        cam_b.get_position(),
        cam_b.get_position() - cam_b.get_all_wcs_points()[0],
    )
    computed = fp_g + vector_gh * 0.5
    return computed, np.linalg.norm(computed - markers[0]), fp_g, fp_h, vector_gh

for correct_distortion in (False, True):
    for cam in cams.values():
        cam.correct_distortion = correct_distortion

    print(f"\n=== correct_distortion = {correct_distortion} ===")
    computed, error, fp_g, fp_h, gh = compute_pair(cams[3], cams[6])
    print(f"Computed:\n{computed}\nError={error}")


plot_single_image_plane_figure(cams[3])
plot_single_image_plane_figure(cams[6])

plt.show()
