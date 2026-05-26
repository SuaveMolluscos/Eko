import numpy as np
from mathematics.vec3 import vec3

class mat4:
    __slots__  = ('data',)
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be ndarray")
        if data.shape != (4, 4):
            raise TypeError("The matrix must be 4x4")
        self.data = data.astype(np.float32, copy=True)

    #Static methods
        #   Crea una matriz identidad
    @staticmethod
    def identity():
        return mat4(np.eye(4, dtype=np.float32))
    
        #Crear una matriz de translacion apartir de (x,y,z)
    @staticmethod
    def translation(x: float=0, y: float=0, z: float=0):
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            raise TypeError("x, y, z must be int or float")
        m = np.eye(4, dtype=np.float32)
        m[0,3] = x
        m[1,3] = y
        m[2,3] = z
        return mat4(m)
    
        #Crea una matriz de escalacion apartir de (x,y,z). default (x=1, y=1, z=1)
    @staticmethod
    def scale(x: float=1, y: float=1, z: float=1):
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            raise TypeError("x, y, z must be int or float")
        m = np.eye(4, dtype=np.float32)
        np.fill_diagonal(m, [x, y, z, 1])
        return mat4(m)
    
        #Crea la matriz de rotacion en x, default en radianes, degrees=False
    @staticmethod
    def rotation_x(angle: float=0, degrees: bool = False):
        if not isinstance(angle, (int, float)):
            raise TypeError("angle must be an integer or a floating-point number")
        if not isinstance(degrees, bool):
            raise TypeError("degrees must be a boolean")
        rad = np.radians(angle) if degrees else angle
        if rad == 0:
            return mat4.identity()
        c, s = np.cos(rad, dtype=np.float32), np.sin(rad, dtype=np.float32)
        m = np.eye(4, dtype=np.float32)
        m[1,1] = c
        m[1,2] = -s
        m[2,1] = s
        m[2,2] = c
        return mat4(m)
    
        #Crea la matriz de rotacion en y, default en radianes, degrees=False
    @staticmethod
    def rotation_y(angle: float=0, degrees: bool = False):
        if not isinstance(angle, (int, float)):
            raise TypeError("angle must be an integer or a floating-point number")
        if not isinstance(degrees, bool):
            raise TypeError("degrees must be a boolean")
        rad = np.radians(angle) if degrees else angle
        if rad == 0:
            return mat4.identity()
        c, s = np.cos(rad, dtype=np.float32), np.sin(rad, dtype=np.float32)
        m = np.eye(4, dtype=np.float32)
        m[0,0] = c
        m[0,2] = s
        m[2,0] = -s
        m[2,2] = c
        return mat4(m)
    
        #Crea la matriz de rotacion en z, default en radianes, degrees=False
    @staticmethod
    def rotation_z(angle: float=0, degrees: bool = False):
        if not isinstance(angle, (int, float)):
            raise TypeError("angle must be an integer or a floating-point number")
        if not isinstance(degrees, bool):
            raise TypeError("degrees must be a boolean")
        rad = np.radians(angle) if degrees else angle
        if rad == 0:
            return mat4.identity()
        c, s = np.cos(rad, dtype=np.float32), np.sin(rad, dtype=np.float32)
        m = np.eye(4, dtype=np.float32)
        m[0,0] = c
        m[0,1] = -s
        m[1,0] = s
        m[1,1] = c
        return mat4(m)
    
        #Matriz vista: from_axes (desde ejes)
    @staticmethod
    def from_axes(cam_position:'vec3'=None, front:'vec3'=None, right:'vec3'=None, up:'vec3'=None):
        if cam_position is None: cam_position = vec3.zero()
        if not isinstance(cam_position, vec3): raise TypeError("cam_position must be a vec3")
        if front is None: front = vec3.forward()
        if not isinstance(front, vec3): raise TypeError("front must be a vec3")
        if right is None: right = vec3.right()
        if not isinstance(right, vec3): raise TypeError("right must be a vec3")
        if up is None: up = vec3.up()
        if not isinstance(up, vec3): raise TypeError("up must be a vec3")

        f = front.normalized()
        r = right.normalized()
        u = up.normalized()

        return mat4(np.array([
            [r.x, r.y, r.z, -(r @ cam_position)],
            [u.x, u.y, u.z, -(u @ cam_position)],
            [f.x, f.y, f.z, -(f @ cam_position)],
            [0, 0, 0, 1]
        ], dtype=np.float32))
    
        #Matriz vista: look_at (desde un objetivo), con posivilidad de rotar la camara
    @staticmethod
    def look_at(cam_position:'vec3'=None, target:'vec3'=None, up_scene:'vec3'=None, roll:'float'=0, degrees: bool = False):
        if cam_position is None: cam_position = vec3.zero()
        if not isinstance(cam_position, vec3): raise TypeError("cam_position must be a vec3")
        if not isinstance(target, vec3): raise TypeError("target must be a vec3")
        if up_scene is None: up_scene = vec3.up()
        if not isinstance(up_scene, vec3): raise TypeError("up_scene must be a vec3")
        if not isinstance(roll, (int, float)): raise TypeError("roll must be an integer or a floating-point number")
        if not isinstance(degrees, bool): raise TypeError("degrees must be a boolean")

        rad = np.radians(roll) if degrees else roll
        s, c = np.sin(rad), np.cos(rad)

        front = target - cam_position
        front = front.normalized()
        right = front.cross(up_scene)
        right = right.normalized()
        up = right.cross(front)
        r = right * c - up * s
        u = right * s + up * c
        return mat4.from_axes(cam_position=cam_position, front=front, right=r, up=u)
    
        #Matriz vista: from_euler (desde radianes/grados)
    @staticmethod
    def from_euler (cam_position:'vec3'=None, yaw:'float'=0, pitch:'float'=0, roll:'float'=0, degrees: bool = False, up_scene=None):
        if cam_position is None: cam_position = vec3.zero()
        if up_scene is None: up_scene = vec3.up()
        if not isinstance(cam_position, vec3): raise TypeError("cam_position must be a vec3")
        if not isinstance(up_scene, vec3): raise TypeError("up_scene must be a vec3")
        if not isinstance(yaw, (int, float)): raise TypeError("yaw must be an integer or a floating-point number")
        if not isinstance(pitch, (int, float)): raise TypeError("pitch must be an integer or a floating-point number")
        if not isinstance(roll, (int, float)): raise TypeError("roll must be an integer or a floating-point number")
        if not isinstance(degrees, bool): raise TypeError("degrees must be a boolean")
        rad_Yaw = np.radians(yaw) if degrees else yaw
        rad_Pitch = np.radians(pitch) if degrees else pitch
        rad_Roll = np.radians(roll) if degrees else roll
        Yc, Ys = np.cos(rad_Yaw), np.sin(rad_Yaw)
        Pc, Ps = np.cos(rad_Pitch), np.sin(rad_Pitch)
        front = vec3(Yc*Pc, Ps, Ys*Pc)
        front = front.normalized()
        right = front.cross(up_scene)
        right = right.normalized()
        up = right.cross(front)
        up = up.normalized()
        Rc, Rs = np.cos(rad_Roll), np.sin(rad_Roll)
        r = right*Rc - up*Rs
        u = right*Rs + up*Rc
        return mat4.from_axes(cam_position=cam_position, front=front, right=r, up=u)
    
        #Matriz 4x4: from_quaternion (desde un quaterinion), posicion en ceros para uso comun
    @staticmethod
    def from_quaternion(q:np.ndarray=None, position:vec3=None):
        if position is None: position = vec3.zero()
        if q is None: q = np.array([[1],[0],[0],[0]], dtype=np.float32)
        if not isinstance(q, np.ndarray): raise TypeError("q must be a ndarray")
        w, x, y, z = q[0][0], q[1][0], q[2][0], q[3][0]
        xx = 2 * x * x
        yy = 2 * y * y
        zz = 2 * z * z
        xy = 2 * x * y
        xz = 2 * x * z
        yz = 2 * y * z
        wx = 2 * w * x
        wy = 2 * w * y
        wz = 2 * w * z

        right = vec3(1 - (yy + zz), xy + wz, xz - wy)
        up = vec3(xy - wz, 1 - (xx + zz), yz + wx)
        front = vec3(xz + wy, yz - wx, 1 - (xx + yy))
        return mat4.from_axes(cam_position=position, front=front, right=right, up=up)

    @staticmethod
    def perspective(fov:float=None, aspect:float=None, far:float=None, near:float=None):
        if fov is None: fov = 90
        if aspect is None: aspect = 1280/720
        if far is None: far = 300
        if near is None: near = 0.1
        if not isinstance(fov, (int, float)): raise TypeError("fov must be an integer or a floating-point number")
        if not isinstance(aspect, (int, float)): raise TypeError("aspect must be an integer or a floating-point number")
        if not isinstance(far, (int, float)): raise TypeError("far must be an integer or a floating-point number")
        if not isinstance(near, (int, float)): raise TypeError("near must be an integer or a floating-point number")
        if near >= far: raise ValueError("near plane must be less than far plane")
        if near <= 0: raise ValueError("near plane must be greater than zero")

        fov_rad = np.radians(fov)
        f = 1/np.tan(fov_rad/2)
        return mat4(np.array([  [f/aspect, 0, 0, 0],
                                [0, f, 0, 0],
                                [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
                                [0, 0, 1, 0]]))
    
        #Reprecentacion en consola
    def __repr__(self):
        matrix_str = str(self.data)
        indented_str = matrix_str.replace('\n', '\n     ')
        return f"mat4({indented_str})"