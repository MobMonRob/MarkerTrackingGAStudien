import numpy as np
import matplotlib.pyplot as plt
import random
import math

"""
Script which was used to implement experimental solving of the undistortion problem according to the Brown-Conrady-model. 
TODO: Cleanup :)
"""

# Distortion coefficients (change these to experiment)
k1 = 1.0526392250425774e-07 
k2 = -1.1849773633625722e-16 
k3 = 6.3492415888196438e-21
#k1 = -0.25    # negative -> barrel, positive -> pincushion
#k2 = 0.05
#k3 = 0.0

# Distortion center (normalized image coordinates: -1 to 1)
x_c, y_c = (1004.8255975935912, 524.48382784691591)
#x_c, y_c = (0, 0)

def distort_points(x_u, y_u):
    dx = x_u - x_c
    dy = y_u - y_c
    r2 = dx*dx + dy*dy
    scale = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    
    x_d = x_c + dx * scale
    y_d = y_c + dy * scale
    return x_d, y_d

# Whatever Chatty cooked here won't work
# def distort_points(x_u, y_u):
#     dx = x_u - x_c
#     dy = y_u - y_c
#     r2 = dx*dx + dy*dy
#     scale = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2

#     x_r = x_u * scale
#     y_r = y_u * scale

#     x_t = 2 * x_c * x_u * y_u + y_c * (r2 + 2 * x_u ** x_u)
#     y_t = 2 * y_c * x_u * y_u + x_c * (r2 + 2 * y_u ** y_u)

#     x_d = x_r + x_t
#     y_d = y_r + y_t

#     return x_d, y_d

def undistort_points(x_d, y_d, iterations = 10):
    dx = x_d - x_c
    dy = y_d - y_c
    r2_hist = []
    for i in range(iterations):
        r2 = dx*dx + dy*dy
        r2_hist.append(r2)
        scale = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2

        dx = (x_d - x_c)/scale
        dy = (y_d - y_c)/scale
        
    x_u = x_c + dx
    y_u = y_c + dy
    
    return x_u, y_u, r2_hist

def undistort_points_newton(x_d, y_d, iterations = 20):
    x_u = x_d
    y_u = y_d

    #r = math.sqrt((x_u - x_c)**2 + (y_u - y_c)**2)
    #f = lambda x_u, x_d, x_c: x_c + (x_d - x_c)/(1 + k1 * ((x_u - x_c)**2 + (y_u - y_c)**2) + k2 * ((x_u - x_c)**2 + (y_u - y_c)**2)**2 + k3 * ((x_u - x_c)**2 + (y_u - y_c)**2)**3) - x_u
    #f_prime = lambda x_u, x_d, x_c: - (x_d - x_c) * ((x_u - x_c) * (2 * k1 + 4 * k2 * r ** 2 + 6 * k3 * r**4))/((1 + k1 * r**2 + k2 * r**4 + k3 * r**6))**2 -1

    f = lambda x_u, x_d, x_c, r: x_c + (x_d - x_c)/(1 + k1 * r**2 + k2 * r**4 + k3 * r**6) - x_u
    f_prime = lambda x_u, x_d, x_c, r: - (x_d - x_c) * ((x_u - x_c) * (2 * k1 + 4 * k2 * r ** 2 + 6 * k3 * r**4))/((1 + k1 * r**2 + k2 * r**4 + k3 * r**6))**2 -1
    
    r_hist = []
    
    for _ in range(iterations):
        r = np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2)
        x_u = x_u - f(x_u, x_d, x_c, r) / f_prime(x_u, x_d, x_c, r)
        y_u = y_u - f(y_u, y_d, y_c, r) / f_prime(y_u, y_d, y_c, r)

        r_hist.append(np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2))
    return x_u, y_u, r_hist

r = lambda: random.choice(range(256))
randomcol = lambda :'#%02X%02X%02X' % (r(),r(),r())

# Generate a grid of points
#grid_range_x = np.linspace(-1024, 1024, 40)
#grid_range_y = np.linspace(-544, 544, 40)
#grid_range_x = np.linspace(-1, 1, 21)
#grid_range_y = np.linspace(-1, 1, 21)
grid_range_x = np.linspace(0, 2048, 40)
grid_range_y = np.linspace(0, 1024, 40)
uu, vv = np.meshgrid(grid_range_x, grid_range_y)

# Distort all points
xd, yd = distort_points(uu, vv)

print('==DEBUG==')
x0, y0 = 0.1, 0.2
x1, y1 = distort_points(x0, y0)
x2, y2, h = undistort_points_newton(x1, y1)
print(x0, y0)
print(x2, y2)

x1, y1 = distort_points(-x0, y0)
x2, y2, h = undistort_points_newton(x1, y1)
print(-x0, y0)
print(x2, y2)

x1, y1 = distort_points(x0, -y0)
x2, y2, h = undistort_points_newton(x1, y1)
print(x0, -y0)
print(x2, y2)

x1, y1 = distort_points(-x0, -y0)
x2, y2, h = undistort_points_newton(x1, y1)
print(-x0, -y0)
print(x2, y2)

print('==DEBUG II==')
x0, y0 = 0.99, 0.99
x1, y1 = distort_points(x0, y0)
x2, y2, h = undistort_points_newton(x1, y1)
print(x0, y0)
print(x2, y2)

x1, y1 = distort_points(-x0, y0)
x2, y2, h = undistort_points_newton(x1, y1)
print(-x0, y0)
print(x2, y2)

x1, y1 = distort_points(x0, -y0)
x2, y2, h = undistort_points_newton(x1, y1)
print(x0, -y0)
print(x2, y2)

x1, y1 = distort_points(-x0, -y0)
x2, y2, h = undistort_points_newton(x1, y1)
print(-x0, -y0)
print(x2, y2)

xu, yu, r2_hist = undistort_points(xd, yd)
xu_newton, yu_newton, r2_hist_newton = undistort_points_newton(xd, yd)

# Plot original vs distorted
fig, ax = plt.subplots(2, figsize=(10,10))
ax[0].plot(uu, vv, color="#36454F", linewidth=0.8)       # original horizontal grid
ax[0].plot(uu.T, vv.T, color="#36454F", linewidth=0.8)   # original vertical grid

ax[0].plot(xd, yd, color="red", linewidth=0.8)        # distorted horizontal grid
ax[0].plot(xd.T, yd.T, color="red", linewidth=0.8)    # distorted vertical grid

ax[0].plot(xu, yu, color="blue", linewidth=0.8)        # undistorted horizontal grid
ax[0].plot(xu.T, yu.T, color="blue", linewidth=0.8)    # undistorted vertical grid

#ax[0].plot(xu_newton, yu_newton, color="green", linewidth=0.8)        # undistorted horizontal grid
#ax[0].plot(xu_newton.T, yu_newton.T, color="green", linewidth=0.8)    # undistorted vertical grid

r2_hist_arr = np.array(r2_hist)       # shape: (iter, 21, 21)
iter_count = r2_hist_arr.shape[0]
r2_lines = r2_hist_arr.reshape(iter_count, -1)
r2_lines = r2_lines.T

i = 0
for curve in r2_lines:
    i += 1
    ax[1].plot(range(iter_count), curve, color=randomcol())
    if i >= 20:
        break
ax[1].set_title("r^2 over iterations")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("r^2")

ax[0].axhline(0, color="black", linewidth=0.5)
ax[0].axvline(0, color="black", linewidth=0.5)
ax[0].set_aspect('equal', 'box')
ax[0].set_title("Forward Radial Distortion Visualization\n(red = distorted, gray = original, blue = undistorted)")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].grid(False)
plt.show()