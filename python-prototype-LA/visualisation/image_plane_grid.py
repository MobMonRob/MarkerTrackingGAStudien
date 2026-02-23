import math
import numpy as np
import matplotlib.pyplot as plt
from model.camera import Camera
from typing import List

def plot_image_plane_grid(cameras: List[Camera]) -> None:
    square = math.ceil(math.sqrt(len(cameras)))

    fig = plt.figure()
    fig.subplots_adjust(left=0.01, bottom=None, right=0.99, top=0.95, wspace=0, hspace=1)

    for camera in cameras:
        ax = fig.add_subplot(square, square, camera.user_id + 1)
        
        ax.set_title(f"UserID {camera.user_id}")
        ax.set_aspect(camera.sensor_xy[1]/camera.sensor_xy[0])
        ax.set(xlim=(0, camera.sensor_xy[0]), ylim=(0, camera.sensor_xy[1]))
        ax.set_xticks([])
        ax.set_yticks([])

        distortionBool = camera.correct_distortion

        camera.correct_distortion=False
        for coordinate in camera.get_centroids():
            ax.scatter(*coordinate, color="#FF0000", s=30, marker='+')
        
        camera.correct_distortion=True
        for coordinate in camera.get_centroids():
            ax.scatter(*coordinate, color="#00FF00", s=30, marker='+')

        camera.correct_distortion = distortionBool

def plot_single_image_plane_figure(camera: Camera) -> None:

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(xlim=(0, camera.sensor_xy[0]), ylim=(0, camera.sensor_xy[1]))
    ax.set_title(f"UserID {camera.user_id}")

    distortionBool = camera.correct_distortion

    camera.correct_distortion=False
    for coordinate in camera.get_centroids():
        ax.scatter(*coordinate, color="#FF0000", s=30, marker='+')
    
    camera.correct_distortion=True
    for coordinate in camera.get_centroids():
        ax.scatter(*coordinate, color="#00FF00", s=30, marker='+')
    
    camera.correct_distortion = distortionBool