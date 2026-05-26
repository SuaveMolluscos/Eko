import math
import numpy as np

class Transform:
    def __init__(self, vertices, translation=(0,0,0), rotation=(0,0,0), scale=(1,1,1)):
        self._vertices    = vertices
        self.translation = list(translation)
        self.rotation    = list(rotation)
        self.scale      = list(scale)

    def get_transformed_vertices(self):
        ex, ey, ez = self.scale
        rx, ry, rz = [math.radians(a) for a in self.rotation]
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        tx, ty, tz = self.translation

        S  = np.array([[ex,0,0,0],[0,ey,0,0],[0,0,ez,0],[0,0,0,1]], dtype=float)
        Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype=float)
        Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=float)
        Rz = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
        T  = np.array([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]], dtype=float)

        M       = T @ Rz @ Ry @ Rx @ S
        verts   = np.array(self._vertices, dtype=float)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        return [tuple(r[:3]) for r in (M @ verts_h.T).T]