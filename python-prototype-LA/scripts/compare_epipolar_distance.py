"""
Comparison: LA vs. PGA epipolar distance.

Uses two cameras from experiment-4 (cameras 6 and 7). Takes one centroid
from camera A, computes its epipolar line in camera B, and measures the
distance from one centroid in B to that line — comparing LA and PGA.
"""

from iotools import camera_config_parser, csv_parser
from model.camera import from_params
from model.geometric_algebra import epipolar_line as el
from model.geometric_algebra import dist_to_line as dtl
from scipy.spatial.transform import Rotation as R
import numpy as np

# --- Setup ---
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

cam_a = cams[CAM_A_ID]
cam_b = cams[CAM_B_ID]

def ga_params(cam):
    wxyz = R.from_matrix(cam.get_rotation_matrix()).as_quat(scalar_first=True)
    pos = cam.get_position().flatten()
    pp = cam.get_principal_point().flatten()
    f = cam.get_focal_length_x()
    return {"f": f, "wxyz": wxyz, "pos": pos, "pp": pp}

pa = ga_params(cam_a)
pb = ga_params(cam_b)

# Use first centroid in A, first centroid in B
centroid_a = cam_a.get_centroids()[0].flatten()
centroid_b = cam_b.get_centroids()[0].flatten()

# --- LA ---
epipole, direction = cam_b.compute_epipolar_line(cam_a, cam_a.get_centroids()[0])
ep = epipole.flatten()
d_la = direction.flatten()
v = centroid_b - ep
lam = np.dot(v, d_la) / np.dot(d_la, d_la)
dist_la = np.linalg.norm(ep + lam * d_la - centroid_b)

# --- PGA ---
line_coeffs = el.epipolar_line(
    a_x=pa["pos"][0], a_y=pa["pos"][1], a_z=pa["pos"][2],
    b_x=pb["pos"][0], b_y=pb["pos"][1], b_z=pb["pos"][2],
    bw_q=pb["wxyz"][0], bx_q=pb["wxyz"][1], by_q=pb["wxyz"][2], bz_q=pb["wxyz"][3],
    cent_x=centroid_a[0], cent_y=centroid_a[1],
    f=pa["f"], f_B=pb["f"],
    p_x=pa["pp"][0], p_y=pa["pp"][1],
    w_q_A=pa["wxyz"][0], x_q_A=pa["wxyz"][1], y_q_A=pa["wxyz"][2], z_q_A=pa["wxyz"][3]
)
le01, le02, le03, le12, le13, le23 = [c[0] for c in line_coeffs]

dist_pga = dtl.dist_to_line(
    cent_x=centroid_b[0], cent_y=centroid_b[1],
    f=pb["f"],
    line_e01=le01, line_e02=le02, line_e03=le03,
    line_e12=le12, line_e13=le13, line_e23=le23,
    pp_x=pb["pp"][0], pp_y=pb["pp"][1]
)[0]

print("=" * 70)
print("Epipolar Distance: LA vs. PGA")
print("=" * 70)
print(f"\n  Camera A: {CAM_A_ID}, centroid 0: ({centroid_a[0]:.2f}, {centroid_a[1]:.2f})")
print(f"  Camera B: {CAM_B_ID}, centroid 0: ({centroid_b[0]:.2f}, {centroid_b[1]:.2f})")
print(f"\n  LA  dist:  {dist_la:.10f} px")
print(f"  PGA dist:  {dist_pga:.10f} px")
print(f"  Δ:         {abs(dist_la - dist_pga):.2e} px")
