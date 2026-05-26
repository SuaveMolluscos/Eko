import sys
import os
import numpy as np

# Añadir el directorio padre al path para poder importar mathematics.vec3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathematics.vec3 import vec3
from mathematics.mat4 import mat4

cam_position = vec3(0, 0, 0)
yaw = 0
pitch = 0
roll = 0
degrees = True
up_scene = vec3(0, 1, 0)

m = mat4.from_euler(cam_position, yaw, pitch, roll, degrees, up_scene)


print(m)