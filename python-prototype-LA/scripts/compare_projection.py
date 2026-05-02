"""
Comparison: LA vs. PGA central projection (3D point → pixel coordinates).

Projects an arbitrary 3D point through two cameras using both implementations
and reports the per-component and Euclidean pixel error.
"""

from iotools import camera_config_parser, csv_parser
from model.camera import from_params
from model.geometric_algebra import central_projection_to_px as cp
from model.geometric_algebra import image_plane as ip
from scipy.spatial.transform import Rotation as R
import numpy as np

XCP_PATH = "../camera-configs/experiment-4.xcp"
CENTROIDS_PATH = "../csvdata/experiment-4/centroid-dump-9-markers-all-cams.csv"

configs = camera_config_parser.parse_file(XCP_PATH)
centroids_raw = csv_parser.load_centroid_data_raw(CENTROIDS_PATH)

CAM_A_ID, CAM_B_ID = 6, 7

cams = {}
for config in configs:
    cam = from_params(config, csv_parser.centroids_for_camera_columns(centroids_raw, config.user_id))
    if cam.user_id in (CAM_A_ID, CAM_B_ID):
        cams[cam.user_id] = cam

test_point = np.array([[200.0], [-150.0], [1500.0]])

print("=" * 70)
print("Central Projection: LA vs. PGA")
print("=" * 70)
print(f"\n  3D point: ({test_point[0,0]:.1f}, {test_point[1,0]:.1f}, {test_point[2,0]:.1f}) mm\n")

for cam_id in (CAM_A_ID, CAM_B_ID):
    cam = cams[cam_id]
    wxyz = R.from_matrix(cam.get_rotation_matrix()).as_quat(scalar_first=True)
    pos = cam.get_position().flatten()
    pp = cam.get_principal_point().flatten()
    f = cam.get_focal_length_x()

    # LA
    px_la = cam.project(test_point).flatten()

    # PGA
    plane = ip.create_image_plane(f, 0, 0, pos[0], pos[1], pos[2], wxyz[0], wxyz[1], wxyz[2], wxyz[3])
    u, v = cp.central_projection_to_px(
        plane[0][0], plane[1][0], plane[2][0],
        pos[0], pos[1], pos[2], plane[3][0],
        pp[0], pp[1],
        test_point[0, 0], test_point[1, 0], test_point[2, 0],
        wxyz[0], wxyz[1], wxyz[2], wxyz[3]
    )
    px_pga = np.array([u[0], v[0]])

    print(f"  Camera {cam_id}:")
    print(f"    LA:    ({px_la[0]:.10f}, {px_la[1]:.10f})")
    print(f"    PGA:   ({px_pga[0]:.10f}, {px_pga[1]:.10f})")
    print(f"Dist: {np.linalg.norm(px_la-px_pga):.2e} px\n")
