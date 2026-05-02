import numpy as np
def central_projection(normal_x, normal_y, normal_z, o_x, o_y, o_z, offset, px, py, pz):
	point_world = np.zeros(16)
	point_world[11] = (-pz) # e0 ^ (e1 ^ e2)
	point_world[13] = (-px) # e0 ^ (e2 ^ e3)
	camera_position = np.zeros(16)
	camera_position[11] = (-o_z) # e0 ^ (e1 ^ e2)
	camera_position[13] = (-o_x) # e0 ^ (e2 ^ e3)
	line_of_sight = np.zeros(16)
	line_of_sight[5] = py * (-camera_position[11]) + (-((-point_world[11]) * o_y)) # e0 ^ e1
	line_of_sight[6] = (-((-point_world[13]) * (-camera_position[11]) + (-((-point_world[11]) * (-camera_position[13]))))) # e0 ^ e2
	line_of_sight[7] = (-point_world[13]) * o_y + (-(py * (-camera_position[13]))) # e0 ^ e3
	line_of_sight[8] = (-camera_position[11]) + point_world[11] # e1 ^ e2
	line_of_sight[9] = (-(o_y + (-py))) # e1 ^ e3
	line_of_sight[10] = (-camera_position[13]) + point_world[13] # e2 ^ e3
	intersection = np.zeros(16)
	intersection[11] = line_of_sight[5] * normal_y + (-(line_of_sight[6] * normal_x)) + line_of_sight[8] * offset # e0 ^ (e1 ^ e2)
	intersection[12] = line_of_sight[5] * normal_z + (-(line_of_sight[7] * normal_x)) + line_of_sight[9] * offset # e0 ^ (e1 ^ e3)
	intersection[13] = line_of_sight[6] * normal_z + (-(line_of_sight[7] * normal_y)) + line_of_sight[10] * offset # e0 ^ (e2 ^ e3)
	intersection[14] = line_of_sight[8] * normal_z + (-(line_of_sight[9] * normal_y)) + line_of_sight[10] * normal_x # e1 ^ (e2 ^ e3)
	intersect_x = np.zeros(16)
	intersect_x[0] = (-intersection[13]) * intersection[14] / (intersection[14] * intersection[14]) # 1.0
	intersect_y = np.zeros(16)
	intersect_y[0] = intersection[12] * intersection[14] / (intersection[14] * intersection[14]) # 1.0
	intersect_z = np.zeros(16)
	intersect_z[0] = (-intersection[11]) * intersection[14] / (intersection[14] * intersection[14]) # 1.0
	return intersect_x, intersect_y, intersect_z
