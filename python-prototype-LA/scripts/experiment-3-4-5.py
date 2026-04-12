from iotools import camera_config_parser, csv_parser
from model.camera import *
import argparse
import numpy as np
from algorithms.lsp import approximate_intersect
import geometry.transforms as transforms
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment

"""
Epsilon search: systematically sweep max_dist values to find one that
produces exactly 9 clusters (one per physical marker) whose LSP-triangulated
positions match the known ground-truth markers.
"""

# Hard-coded paths
XCP_PATH = "../../camera-configs/experiment-4.xcp"
CENTROIDS_PATH = "../../csvdata/experiment-4/centroid-dump-9-markers-all-cams.csv"
MARKERS_PATH = "../../csvdata/experiment-4/marker-dump-9-markers-all-cams.csv"

# Load data
configs = camera_config_parser.parse_file(XCP_PATH)
centroids_raw = csv_parser.load_centroid_data_raw(CENTROIDS_PATH)
markers = csv_parser.load_known_markers(MARKERS_PATH)

SAVE_FIGURES = False

known_pts = np.hstack(markers)  # (3, 9) – each column is a known marker position

cams_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
cams = {}
for config in configs:
    cam = from_params(config, csv_parser.centroids_for_camera_columns(centroids_raw, config.user_id))
    if cam.user_id in cams_of_interest:
        cams[str(cam.user_id)] = cam


# ── helper: run one trial for a given max_dist ──────────────────────────
def run_trial(max_dist: float):
    """
    Returns (n_clusters, total_distance) where total_distance is the sum of
    the optimal (Hungarian) assignment between triangulated cluster centres
    and known markers.  Returns (n_clusters, np.inf) when triangulation is not
    possible for every cluster.
    """
    # Reset correspondences
    for cam in cams.values():
        cam.init_correspondences()

    # Compute correspondences
    for cam in cams.values():
        for other in cams.values():
            if other.user_id == cam.user_id:
                continue
            cam.compute_correspondences_with(other, max_dist=max_dist)

    # Build directed graph
    G = nx.DiGraph()
    for cam in cams.values():
        for idx, corrs in cam.correspondences.items():
            node_a = (cam.user_id, idx)
            for (other_id, other_idx, dist) in corrs:
                G.add_edge(node_a, (other_id, other_idx), weight=dist)

    components = list(nx.weakly_connected_components(G))
    n_clusters = len(components)

    # Triangulate each cluster via LSP
    computed = []
    for comp in components:
        pts, dirs = [], []
        for (cam_id, cent_idx) in comp:
            c = cams[str(cam_id)]
            centroid = c.get_centroids_map()[cent_idx]
            p_ccs = c.image_to_ccs(centroid)
            p_wcs = transforms.dehomogenize(c.ccs_to_wcs(p_ccs))
            pts.append(c.get_position())
            dirs.append(p_wcs - c.get_position())
        if len(pts) >= 2:
            computed.append(approximate_intersect(pts, dirs))
        else:
            return n_clusters, np.inf  # can't triangulate a singleton

    # Optimal assignment (Hungarian) between computed and known markers
    computed_mtx = np.hstack(computed)  # (3, n_clusters)
    cost = np.zeros((n_clusters, len(markers)))
    for i in range(n_clusters):
        for j in range(len(markers)):
            cost[i, j] = np.linalg.norm(computed_mtx[:, i] - known_pts[:, j])
    row_ind, col_ind = linear_sum_assignment(cost)
    total_dist = cost[row_ind, col_ind].sum()
    return n_clusters, total_dist


# ── sweep ────────────────────────────────────────────────────────────────
epsilons = np.arange(0.01, 1.01, 0.005)

results_eps = []
results_nclusters = []
results_dist = []
results_hit_nine = []

for eps in epsilons:
    n_cl, tot_d = run_trial(eps)
    results_eps.append(eps)
    results_nclusters.append(n_cl)
    results_dist.append(tot_d)
    results_hit_nine.append(n_cl == 9)
    tag = "✓" if n_cl == 9 else "✗"
    print(f"max_dist={eps:7.2f}  clusters={n_cl:3d}  Σdist={tot_d:10.1f}  {tag}")

results_eps = np.array(results_eps)
results_dist = np.array(results_dist)
results_nclusters = np.array(results_nclusters)
results_hit_nine = np.array(results_hit_nine)

# ── plot ─────────────────────────────────────────────────────────────────
fig, (ax_dist, ax_cl) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: sum of distances
mask9 = results_hit_nine
mask_not9 = ~results_hit_nine

ax_dist.plot(results_eps[mask9], results_dist[mask9], 'o-', color='steelblue',
             label='9 clusters (sum of marker distances)')
ax_dist.plot(results_eps[mask_not9], results_dist[mask_not9], 'x', color='red',
             markersize=10, markeredgewidth=2, label='≠ 9 clusters')
ax_dist.set_ylabel('Σ known-computed marker distance (mm)')
ax_dist.set_title('Epsilon search — quality of correspondence clustering')
ax_dist.legend()
ax_dist.grid(True, alpha=0.3)

# Bottom: number of clusters
ax_cl.plot(results_eps, results_nclusters, 's-', color='grey', markersize=4)
ax_cl.axhline(9, color='green', linestyle='--', linewidth=1, label='target = 9')
ax_cl.set_xlabel('max_dist (epipolar distance threshold)')
ax_cl.set_ylabel('number of clusters')
ax_cl.legend()
ax_cl.grid(True, alpha=0.3)

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig("epsilon_search.svg", format="svg", dpi=600, transparent=True,
                bbox_inches="tight", pad_inches=0)

# Report best
if mask9.any():
    best_idx = np.argmin(results_dist[mask9])
    best_eps = results_eps[mask9][best_idx]
    best_dist = results_dist[mask9][best_idx]
    print(f"\n★ Best epsilon with 9 clusters: {best_eps:.2f}  (Σdist = {best_dist:.1f} mm)")

    # ── Visualize best attempt in 3D ─────────────────────────────────────
    from visualisation.scene import create_3d_figure, plot_camera_grey, plot_cluster_rays, scatter_3d_markers

    # Re-run the best trial to recover the graph and triangulated positions
    for cam in cams.values():
        cam.init_correspondences()
    for cam in cams.values():
        for other in cams.values():
            if other.user_id == cam.user_id:
                continue
            cam.compute_correspondences_with(other, max_dist=best_eps)

    G_best = nx.DiGraph()
    for cam in cams.values():
        for idx, corrs in cam.correspondences.items():
            node_a = (cam.user_id, idx)
            for (other_id, other_idx, dist) in corrs:
                G_best.add_edge(node_a, (other_id, other_idx), weight=dist)

    best_components = [list(comp) for comp in nx.weakly_connected_components(G_best)]
    best_positions = []
    for comp in best_components:
        pts, dirs = [], []
        for (cam_id, cent_idx) in comp:
            c = cams[str(cam_id)]
            centroid = c.get_centroids_map()[cent_idx]
            p_ccs = c.image_to_ccs(centroid)
            p_wcs = transforms.dehomogenize(c.ccs_to_wcs(p_ccs))
            pts.append(c.get_position())
            dirs.append(p_wcs - c.get_position())
        if len(pts) >= 2:
            best_positions.append(approximate_intersect(pts, dirs))
        else:
            best_positions.append(None)

    fig3d, ax3d = create_3d_figure(cameras=cams.values(), view_coordinates=False)
    for cam in cams.values():
        plot_camera_grey(ax3d, cam)
    plot_cluster_rays(ax3d, cams, best_components, best_positions)
    scatter_3d_markers(ax3d, markers, size=50, color='green', symbol='*')
    ax3d.set_title(f"Best epsilon = {best_eps:.2f}  —  {len(best_components)} clusters  —  Σdist = {best_dist:.1f} mm")

    from matplotlib.lines import Line2D
    legend_computed = Line2D([], [], color='black', marker='x', linestyle='None',
                             markersize=8, label='Computed (LSP triangulation)')
    legend_known = Line2D([], [], color='green', marker='*', linestyle='None',
                          markersize=10, label='Known (Vicon API)')
    ax3d.legend(handles=[legend_computed, legend_known], loc='upper left', fontsize=9)

    if SAVE_FIGURES:
        fig3d.savefig("epsilon_best_3d.svg", format="svg", dpi=600, transparent=True,
                    bbox_inches="tight", pad_inches=0)
else:
    print("\n✗ No epsilon produced exactly 9 clusters in the tested range.")

plt.show()