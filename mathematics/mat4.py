import numpy as np
import math
from mathematics.vec3 import vec3

class mat4:

    __slots__  = ('data',)

        #Costructor
    def __init__(self, data):
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be ndarray or list/tuple")
        if data.shape != (4, 4):
            raise ValueError("The matrix must be 4x4")
        self.data = data.astype(np.float32, copy=True)

        #Reprecentacion
    def __repr__(self):
        st = str(self.data)
        st = st.replace('\n', '\n     ')
        return f"mat4( {st} )"
    
        #Crea un Mat4 - identidad
    @classmethod
    def identity(cls):
        return cls(np.eye(4, dtype=np.float32))
    
        #Crea un Mat4 - translacion apartir de (x,y,z)
    @classmethod
    def translation(cls, v, y=None, z=None):
        if isinstance(v, vec3):
            x, y, z = v.x, v.y, v.z
        elif all(isinstance(i, (int, float)) for i in (v,y,z)):
            x, y, z = v, y, z
        else:
            raise TypeError("translation requires a vec3 or three numeric values")
        m = np.eye(4, dtype=np.float32)
        m[0,3], m[1,3], m[2,3] = x, y, z
        return cls(m)
    
        #Crea un Mat4 - escalacion apartir de (x,y,z)
    @classmethod
    def scale(cls, v, y=None, z=None):
        if isinstance(v, vec3):
            x, y, z = v.x, v.y, v.z
        elif all(isinstance(i ,(int, float)) for i in (v,y,z)):
            x, y, z = v, y, z
        else:
            raise TypeError("scale requires a vec3 or three numeric values")
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[1,1], m[2,2] = x, y, z
        return cls(m)
    
        #Crea un Mat4 - Rotacion eje Z
    @classmethod
    def rotation_z(cls, angle, from_degrees: bool = False):
        if not isinstance(angle, (int, float)):
            raise TypeError("angle must be numeric (int or float)")
        if not isinstance(from_degrees, bool):
            raise TypeError("from_degrees must be bool (True or False)")
        if from_degrees: angle = math.radians(angle)
        s, c = math.sin(angle), math.cos(angle)
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[0,1], m[1,0], m[1,1] = c, -s, s, c
        return cls(m)

        #Crea un Mat4 - Rotacion eje Y
    @classmethod
    def rotation_y(cls, angle, from_degrees: bool = False):
        if not isinstance(angle, (int, float)):
            raise TypeError("angle must be numeric (int or float)")
        if not isinstance(from_degrees, bool):
            raise TypeError("from_degrees must be bool (True or False)")
        if from_degrees: angle = math.radians(angle)
        s, c = math.sin(angle), math.cos(angle)
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[0,2], m[2,0], m[2,2] = c, s, -s, c
        return cls(m)

        #Crea un Mat4 - Rotacion eje X
    @classmethod
    def rotation_x(cls, angle, from_degrees: bool = False):
        if not isinstance(angle, (int, float)):
            raise TypeError("angle must be numeric (int or float)")
        if not isinstance(from_degrees, bool):
            raise TypeError("from_degrees must be bool (True or False)")
        if from_degrees: angle = math.radians(angle)
        s, c = math.sin(angle), math.cos(angle)
        m = np.eye(4, dtype=np.float32)
        m[1,1], m[1,2], m[2,1], m[2,2] = c, -s, s, c
        return cls(m)
    
        #Matrices de vista
    @classmethod
    def from_axes(cls, position: vec3=None, front: vec3=None, right: vec3=None, up: vec3=None):
        if position is None: position = vec3.zero()
        if front is None: front = vec3.forward()
        if right is None: right = vec3.right()
        if up is None: up = vec3.up()
        for name, var in zip(("position", "front", "right", "up"), (position, front, right, up)):
            if not isinstance(var, vec3):
                raise TypeError(f"{name} must be a vec3")
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[0,1], m[0,2], m[0,3] = right.x, right.y, right.z, -(right @ position)
        m[1,0], m[1,1], m[1,2], m[1,3] = up.x, up.y, up.z, -(up @ position)
        m[2,0], m[2,1], m[2,2], m[2,3] = -front.x, -front.y, -front.z, front @ position
        return cls(m)
    
    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3, roll: float=0.0, from_degrees: bool = False):
        f = (target - position).normalized()
        r = (f.cross(up)).normalized()
        u = r.cross(f)
        if roll != 0.0:
            if from_degrees: roll = math.radians(roll)
            sr, cr = math.sin(roll), math.cos(roll)
            r, u = r * cr + u * sr, -r * sr + u * cr
        return cls.from_axes(position=position, front=f, right=r, up=u)
    
    @classmethod
    def from_euler(cls, position: vec3, yaw: float, pitch: float, roll: float, up_scene: vec3=None, from_degrees: bool = False):
        if up_scene is None: up_scene = vec3.up()
        if from_degrees:
            yaw, pitch = math.radians(yaw), math.radians(pitch)
        f = vec3(math.cos(yaw) * math.cos(pitch), math.sin(pitch), math.sin(yaw) * math.cos(pitch)).normalized()
        r = (f.cross(up_scene)).normalized()
        u = (r.cross(f)).normalized()
        if roll != 0:
            if from_degrees: roll = math.radians(roll)
            cr, sr = math.cos(roll), math.sin(roll)
            r, u = r*cr-u*sr, r*sr+u*cr
        return cls.from_axes(position=position, front=f, right=r, up=u)
    
    @classmethod
    def from_quaternion(cls, q:np.ndarray=None, position:vec3=None):
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
        return cls.from_axes(position=position, front=front, right=right, up=up)
    
    @classmethod
    def perspective(cls, fov: float=1.57, aspect: float=1280/720, near:float=0.1, far=300, from_degrees: bool=False):
        if from_degrees: fov = math.radians(fov)
        f = 1.0/math.tan(fov/2.0)
        m = np.zeros((4,4), dtype=np.float32)
        m[0,0], m[1,1] = f/aspect, f
        m[2,2],m[2,3] = (far+near)/(near-far), 2*far*near/(near-far)
        m[3,2] = -1
        return cls(m)
    
    @classmethod
    def orthographic(cls, left: float=0, right: float=1280, bottom: float=720, top: float=0, near:float=0.1, far:float=300):
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[0,3] = 2/(right-left), -((right+left)/(right-left))
        m[1,1], m[1,3] = 2/(top-bottom), -((top+bottom)/(top-bottom))
        m[2,2], m[2,3] = -(2/(far-near)), -((far+near)/(far-near))
        return cls(m)
    
    @classmethod
    def rotation_from_quaternion(cls, q: np.ndarray):
        if q is None: q = np.array([1,0,0,0], dtype=np.float32)
        if not isinstance(q, np.ndarray): raise TypeError("q must be ndarray")
        w, x, y, z = q.flatten()
        xx, yy, zz = 2*x*x, 2*y*y, 2*z*z
        xy, xz, yz = 2*x*y, 2*x*z, 2*y*z
        wx, wy, wz = 2*w*x, 2*w*y, 2*w*z
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[0,1], m[0,2] = 1-(yy+zz), xy+wz, xz-wy
        m[1,0], m[1,1], m[1,2] = xy-wz, 1-(xx+zz), yz+wx
        m[2,0], m[2,1], m[2,2] = xz+wy, yz-wx, 1-(xx+yy)
        return cls(m)
    
    def __matmul__(self, other):
        if isinstance(other, mat4):
            return mat4(self.data @ other.data)
        elif isinstance(other, vec3):
            return vec3.from_numpy(self.data @ other.to_homogeneous())
        elif isinstance(other, np.ndarray):
            return self.data @ other
        elif isinstance(other, (int, float)):
            return mat4(self.data*other)
        else:
            raise TypeError("Unsupported type for mat4 multiplication")
        
    def transform_points_batch(self, batch, w: float = 1.0):
        if isinstance(batch, np.ndarray):
            ws   = np.full((len(batch), 1), w, dtype=np.float32)
            v4   = np.hstack([batch, ws])
            res  = (self.data @ v4.T).T
            return res[:, :3]
        elif isinstance(batch, (list, tuple)) and isinstance(batch[0], vec3):
            arr = np.array([[v.x, v.y, v.z] for v in batch], dtype=np.float32)
            return self.transform_points_batch(arr, w=w)
        else:
            raise TypeError("batch must be ndarray (N,3) or list/tuple of vec3")

    def inverse(self):
        try:
            return mat4(np.linalg.inv(self.data))
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not invertible")
        
    def transpose(self):
        return mat4(self.data.T)
    
    def normal_matrix(self):
        return self.inverse().transpose()
    
    def fast_inverse(self):
        M = self.data
        t = M[:3, 3]
        RS = M[:3, :3]
        sx = np.linalg.norm(RS[:,0])
        sy = np.linalg.norm(RS[:,1])
        sz = np.linalg.norm(RS[:,2])
        R = np.array([
            RS[:,0]/sx,
            RS[:,1]/sy,
            RS[:,2]/sz
        ]).T
        S_inv = np.diag([1/sx, 1/sy, 1/sz])
        R_inv = R.T
        T_inv = -R_inv @ S_inv @ t
        M_inv = np.eye(4, dtype=np.float32)
        M_inv[:3,:3] = R_inv @ S_inv
        M_inv[:3, 3] = T_inv
        return mat4(M_inv)
    
    def get_translation(self):
        return vec3(self.data[0,3], self.data[1,3], self.data[2,3])
    
    def get_scale(self):
        sx = np.linalg.norm(self.data[:3,0])
        sy = np.linalg.norm(self.data[:3,1])
        sz = np.linalg.norm(self.data[:3,2])
        return vec3(sx, sy, sz)
    
    @staticmethod
    def quaternion_from_matrix(R: np.ndarray):
        tr = R[0,0] + R[1,1] + R[2,2]

        if tr > 0:
            w = 0.5 * np.sqrt(1.0 + tr)
            x = (R[2,1] - R[1,2]) / (4.0*w)
            y = (R[0,2] - R[2,0]) / (4.0*w)
            z = (R[1,0] - R[0,1]) / (4.0*w)
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                x = 0.5 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                w = (R[2,1] - R[1,2]) / (4.0*x)
                y = (R[0,1] + R[1,0]) / (4.0*x)
                z = (R[0,2] + R[2,0]) / (4.0*x)
            elif R[1,1] > R[2,2]:
                y = 0.5 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                w = (R[0,2] - R[2,0]) / (4.0*y)
                x = (R[0,1] + R[1,0]) / (4.0*y)
                z = (R[1,2] + R[2,1]) / (4.0*y)
            else:
                z = 0.5 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                w = (R[1,0] - R[0,1]) / (4.0*z)
                x = (R[0,2] + R[2,0]) / (4.0*z)
                y = (R[1,2] + R[2,1]) / (4.0*z)

        return np.array([w, x, y, z], dtype=np.float32)
    
    def decompose(self):
        M = self.data

        pos = vec3(M[0,3], M[1,3], M[2,3])

        sx = np.linalg.norm(M[:3,0])
        sy = np.linalg.norm(M[:3,1])
        sz = np.linalg.norm(M[:3,2])
        scale = vec3(sx, sy, sz)

        R = np.zeros((3,3), dtype=np.float32)
        R[:,0] = M[:3,0] / sx
        R[:,1] = M[:3,1] / sy
        R[:,2] = M[:3,2] / sz

        quat = mat4.quaternion_from_matrix(R)

        return pos, quat, scale
    
    def det(self):
        return float(np.linalg.det(self.data))
    
    def lerp(self, other, t):
        return mat4(self.data * (1-t) + other.data * t)
    
    def __eq__(self, value, tol=1e-6):
        return np.allclose(self.data, value.data, atol=tol)