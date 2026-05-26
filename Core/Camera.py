import math
import numpy as np

def cross(A, B):
    return (A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0])

def normalize(cr):
    magnitude = math.sqrt((cr[0]**2 + cr[1]**2 + cr[2]**2))
    if magnitude == 0:
        return (0,0,0)
    return (cr[0]/magnitude, cr[1]/magnitude, cr[2]/magnitude)

def dot(A,B):
    return A[0]*B[0] + A[1]*B[1] + A[2]*B[2]

class Camera:
    def __init__(self, Position=(0,0,0), rotation=(0,0,0), fov=60, near=0.1, far=300, width=640, height=480):
        self.Position = list(Position)
        self.Yaw = rotation[0]
        self.Pitch = rotation[1]
        self.Roll = rotation[2]
        self.Fov = fov
        self.Near = near
        self.Far = far
        self.width = width
        self.height = height
        self.Front = (0.0, 0.0, 1.0)
        self.Right = (1.0, 0.0, 0.0)
        self.Up    = (0.0, 1.0, 0.0)
        self.WorldUp = (0, 1, 0)
    
    def view_matrix(self):
        pitch = max(-89.0, min(89.0, self.Pitch))

        Ryaw = math.radians(self.Yaw)
        Rpitch = math.radians(pitch)
        Rroll = math.radians(self.Roll)

        Fx = math.cos(Ryaw) * math.cos(Rpitch)
        Fy = math.sin(Rpitch)
        Fz = math.sin(Ryaw) * math.cos(Rpitch)
        Front = normalize((Fx, Fy, Fz))

        up_reference = self.WorldUp
        cross_right = cross(Front, up_reference)
        cross_magnitude = math.sqrt(cross_right[0] ** 2 + cross_right[1] ** 2 + cross_right[2] ** 2)

        if cross_magnitude < 1e-8:
            up_reference = (0, 0, 1)
            cross_right = cross(Front, up_reference)

        Right = normalize(cross_right)
        Up = normalize(cross(Right, Front))

        cr, sr = math.cos(Rroll), math.sin(Rroll)
        _Right = (
            Right[0] * cr - Up[0] * sr,
            Right[1] * cr - Up[1] * sr,
            Right[2] * cr - Up[2] * sr,
        )
        _Up = (
            Right[0] * sr + Up[0] * cr,
            Right[1] * sr + Up[1] * cr,
            Right[2] * sr + Up[2] * cr,
        )

        Right = normalize(_Right)
        Up = normalize(_Up)

        self.Front = Front
        self.Right = Right
        self.Up    = Up

        return np.array([[Right[0], Right[1], Right[2], -dot(Right, self.Position)],
                        [Up[0], Up[1], Up[2], -dot(Up, self.Position)],
                        [-Front[0], -Front[1], -Front[2], dot(Front, self.Position)],
                        [0, 0, 0, 1]])

    def projection_matrix(self):
        f = 1 / math.tan(math.radians(self.Fov) / 2)
        aspect = self.width/self.height
        return np.array([[f/aspect, 0, 0, 0],
                        [0, f, 0, 0],
                        [0, 0, (self.Far+self.Near)/(self.Near-self.Far), 2*self.Far*self.Near/(self.Near-self.Far)],
                        [0, 0, -1, 0]])