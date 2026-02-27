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
- xcp = camera-configs/recalibrated-config.xcp
- markers =  csvdata/recalibrated/marker-dump-1-marker-cam-4-and-7.csv
- centroids = csvdata/recalibrated/centroid-dump-1-marker-cam-4-and-7.csv
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
    # cams.add(from_params(config, csv_parser.centroids_for_camera_columns(centroids_raw, config.user_id)))

# visualisation of observed centroids on image plane
plot_image_plane_grid(cams.values())
plot_single_image_plane_figure(cams["6"])

# visualisation of 3D scene
fig, ax = scene.create_3d_figure()
for camera in cams.values():
    wcs_points = camera.get_all_wcs_points()
    scene.plot_camera(ax, camera)

# Render the dumped points "Musterlösung"
scene.scatter_3d_markers(ax, markers, color="green")

# Render the computed points

cam3 = cams["3"]
cam6 = cams["6"]

#print(f"Position {cam3.get_position()}, WCS POINT {cam6.get_position()}")

orthogonal_distance_vector, footpoint_g, footpoint_h = lotfusspunkt_verfahren(cam3.get_position(), 
                                                          cam3.get_position() - cam3.get_all_wcs_points()[0],
                                                          cam6.get_position(), 
                                                          cam6.get_position() - cam6.get_all_wcs_points()[0])


print(f'---------------')                   
computed_marker = footpoint_g + orthogonal_distance_vector*0.5
print(f'Computed Marker Position: {computed_marker}')
print(f'Actual Marker Position: {markers[0]}')
#scene.scatter_3d_markers(ax, [footpoint_g, footpoint_h], color="pink")
scene.scatter_3d_markers(ax, [footpoint_g + orthogonal_distance_vector*0.5])

print(f'Error between computed and actual marker: {np.linalg.norm(computed_marker - markers[0])}')

ray = ray_batch_from_to(footpoint_g, footpoint_h)
ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color="pink", linestyle='dashed')

plt.show()