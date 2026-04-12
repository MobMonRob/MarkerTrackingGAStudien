from iotools import camera_config_parser, csv_parser
from model.camera import *
import argparse
import pandas as pd
import numpy as np
from visualisation.image_plane_grid import plot_single_image_plane_figure, plot_epipolar_line
from visualisation.scene import create_3d_figure, plot_camera, scatter_3d_markers, plot_camera_grey, plot_cluster_rays
import matplotlib.pyplot as plt
import networkx as nx

# Hard-coded paths
XCP_PATH = "../../camera-configs/experiment-4.xcp"
CENTROIDS_PATH = "../../csvdata/experiment-4/centroid-dump-9-markers-all-cams.csv"
MARKERS_PATH = "../../csvdata/experiment-4/marker-dump-9-markers-all-cams.csv"

# Load data
configs = camera_config_parser.parse_file(XCP_PATH)
centroids_raw = csv_parser.load_centroid_data_raw(CENTROIDS_PATH)
markers = csv_parser.load_known_markers(MARKERS_PATH)

SAVE_FIGURES = False

#Epsilon-Threshold
MAX_DIST = float('inf')

#Abb 13 & 14
#cams_of_interest = [11, 6]

#Abb 15 & 16
cams_of_interest = [6, 7]

# instantiate camera objects
cams = {}
for config in configs:
    new_camera = from_params(config, csv_parser.centroids_for_camera_columns(centroids_raw, config.user_id))
    if new_camera.user_id in cams_of_interest:
        cams[str(new_camera.user_id)] = new_camera

cam_plots = {}
fig3d, ax3d = create_3d_figure(cameras=cams.values(), view_coordinates=False)
for cam in cams.values():
    fig, ax = plot_single_image_plane_figure(cam)
    cam_plots[cam.user_id] = (fig, ax)

    plot_camera(ax3d, cam, cameras_total=13)
scatter_3d_markers(ax3d, markers, size=50, color="green", symbol="*")


for cam in cams.values():
    for other_cam in cams.values():
        if other_cam.user_id == cam.user_id:
            continue
        for centroid in other_cam.get_centroids():
            plot_epipolar_line(cam, other_cam, centroid, cam_plots[cam.user_id][1])

# Compute correspondences and build directed graph
for cam in cams.values():
    for other_cam in cams.values():
        if other_cam.user_id == cam.user_id:
            continue
        cam.compute_correspondences_with(other_cam, max_dist=MAX_DIST)

G = nx.DiGraph()
for cam in cams.values():
    for idx, corrs in cam.correspondences.items():
        node_a = (cam.user_id, idx)
        for (other_cam_id, other_idx, dist) in corrs:
            G.add_edge(node_a, (other_cam_id, other_idx), weight=dist)

# Color by cluster
components = list(nx.weakly_connected_components(G))
cmap = plt.cm.tab20
node_color_map = {}
for i, comp in enumerate(components):
    for node in comp:
        node_color_map[node] = cmap(i % 20)
node_colors = [node_color_map[n] for n in G.nodes()]

fig_graph, ax_graph = plt.subplots(figsize=(20, 10))
ax_graph.set_title(f"Correspondence graph — {len(components)} clusters")
pos = nx.nx_agraph.graphviz_layout(G, prog='neato', args='-Goverlap=false -Gsep=+30 -Gsplines=true')
nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=2000, node_color=node_colors)
nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=11, font_weight='bold',
                        labels={n: f"cam{n[0]}:cent{n[1]}" for n in G.nodes()})
# Split edges: bidirectional (both directions exist) get a curve, unidirectional stay straight
mutual_edges = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
one_way_edges = [(u, v) for u, v in G.edges() if not G.has_edge(v, u)]

nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=mutual_edges,
                       arrows=True, arrowsize=25, node_size=2000,
                       connectionstyle="arc3,rad=0.15", alpha=0.8)
nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=one_way_edges,
                       arrows=True, arrowsize=25, node_size=2000,
                       connectionstyle="arc3,rad=0.0", alpha=0.8)

def _arc_midpoint(p0, p1, rad):
    """Midpoint of the quadratic Bezier arc drawn by connectionstyle arc3,rad=rad."""
    mid = (p0 + p1) / 2
    if rad == 0.0:
        return mid
    d = p1 - p0
    dist = np.linalg.norm(d)
    if dist < 1e-10:
        return mid
    unit = d / dist
    perp = np.array([-unit[1], unit[0]])          # 90° CCW
    # bezier midpoint = straight midpoint + 0.5 * rad * dist * perp
    return mid + 0.5 * rad * dist * perp

for u, v, data in G.edges(data=True):
    rad = 0.15 if G.has_edge(v, u) else 0.0
    p0 = np.array(pos[u], dtype=float)
    p1 = np.array(pos[v], dtype=float)
    lx, ly = _arc_midpoint(p0, p1, rad)
    ax_graph.text(lx, ly, f"{data['weight']:.2f}",
                  fontsize=10, ha='center', va='center', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0))

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

legend_node = mpatches.Patch(facecolor='lightgrey', edgecolor='black',
                              label=None)
legend_edge = mlines.Line2D([], [], color='black', linewidth=1.5,
                             marker='>', markersize=8, markerfacecolor='black',
                             label=None)

legend = ax_graph.legend(
    handles=[legend_edge],
    labels=[
        'A:centX → B:centY  —  epipolar line of B:centY in A was closest to A:centX',
    ],
    loc='lower left', fontsize=10, framealpha=0.9,
    handlelength=1.5, handleheight=1.2,
)
for text in legend.get_texts():
    text.set_fontweight('bold')

# --- LSP triangulation per cluster ---
from algorithms.lsp import approximate_intersect
import geometry.transforms as transforms

fig3d_lsp, ax3d_lsp = create_3d_figure(cameras=cams.values(), view_coordinates=False)
ax3d_lsp.set_title(f"Scene showing results of LSP triangulation per cluster")
for cam in cams.values():
    plot_camera_grey(ax3d_lsp, cam)

# Build a list of weakly-connected components indexed by (cam_id, centroid_idx) nodes
graph_components = [list(comp) for comp in nx.weakly_connected_components(G)]

triangulated_positions = []
for comp in graph_components:
    ray_points = []
    ray_directions = []
    for (cam_id, centroid_idx) in comp:
        cam = cams[str(cam_id)]
        centroid = cam.get_centroids_map()[centroid_idx]
        point_ccs = cam.image_to_ccs(centroid)
        point_wcs = transforms.dehomogenize(cam.ccs_to_wcs(point_ccs))
        origin = cam.get_position()                        # (3,1)
        direction = point_wcs - origin                     # (3,1)
        ray_points.append(origin)
        ray_directions.append(direction)
    if len(ray_points) >= 2:
        pos_3d = approximate_intersect(ray_points, ray_directions)
    else:
        pos_3d = None
    triangulated_positions.append(pos_3d)

plot_cluster_rays(ax3d_lsp, cams, graph_components, triangulated_positions)
scatter_3d_markers(ax3d_lsp, markers, size=50, color='green', symbol='*')


if(SAVE_FIGURES):
    fig3d_lsp.savefig("figure_lsp.svg", format="svg", dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)

    fig3d.savefig("scene.svg", format="svg", dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
    
    for cam_id in cams_of_interest:
        cam_plots[cam_id][0].savefig(f"cam{cam_id}.svg", format="svg", dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
    
    fig_graph.savefig("correspondence_graph.svg", format="svg", dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)

plt.show()