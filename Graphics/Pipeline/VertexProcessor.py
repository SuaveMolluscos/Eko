import numpy as np

def homogeneous_conversion(x, y, z):
    return np.array( [[x], [y], [z], [1] ] )

def view_to_proyeccion(projection_matrix, homogeneous_coordinate):
    projection = projection_matrix @ homogeneous_coordinate
    w = projection[3][0]
    if w <= 0.1:
        return None
    return projection / w