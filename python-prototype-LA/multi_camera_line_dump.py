from iotools import camera_config_parser, csv_parser
from model.camera import *
from visualisation.image_plane_grid import plot_image_plane_grid, plot_single_image_plane_figure
import visualisation.scene as scene
import matplotlib.pyplot as plt
from algorithms.lvp import lotfusspunkt_verfahren
from geometry.rays import ray_batch_from_to
import argparse

"""
Script intendet to be run with:
- xcp = camera-configs/experiment-1.xcp
- markers =  csvdata/experiment-1/marker-dump-1-marker-all-cams.csv
- centroids = csvdata/experiment-1/centroid-dump-1-marker-all-cams.csv 
"""

parser = argparse.ArgumentParser("playground")
parser.add_argument("-xcp", help="The path to the vicon .xcp config.", type=str)
parser.add_argument("-centroids", help="The path to the centroids csv file.", type=str)
parser.add_argument("-markers", help="The path to the markers csv file dumped via the vicon API.", type=str)

# load the raw data
args = parser.parse_args()
configs = camera_config_parser.parse_file(args.xcp)
centroids_raw = csv_parser.load_centroid_data_raw(args.centroids)
markers = csv_parser.load_known_markers(args.markers)

# instanciate camera objects
cams = {}
for config in configs:
    new_camera = from_params(config, csv_parser.centroids_for_camera_columns(centroids_raw, config.user_id))
    cams[str(new_camera.user_id)] = new_camera

# visualisation of 3D scene, getting only those cameras such that 4 lines intersect at the point
fig, ax = scene.create_3d_figure()
for camera in [cams["3"], cams["6"], cams["4"], cams["5"]]:
    wcs_points = camera.get_all_wcs_points()
    scene.plot_camera(ax, camera)

# Render the dumped points "Musterlösung"
scene.scatter_3d_markers(ax, markers, color="green")

plt.show()