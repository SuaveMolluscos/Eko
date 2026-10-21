import numpy as np
import math
from numbers import Real
from eko.mathematics.vec3 import vec3

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
        elif all(isinstance(i, Real) and not isinstance(i, bool) for i in (v, y, z)):
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
        elif all(isinstance(i, Real) and not isinstance(i, bool) for i in (v, y, z)):
            x, y, z = v, y, z
        else:
            raise TypeError("scale requires a vec3 or three numeric values")
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[1,1], m[2,2] = x, y, z
        return cls(m)
    
        #Crea un Mat4 - Rotacion eje Z
    @classmethod
    def rotation_z(cls, angle, from_degrees: bool = False):
        if not isinstance(angle, Real) or isinstance(angle, bool):
            raise TypeError("angle must be numeric (int or float)")
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
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
        if not isinstance(angle, Real) or isinstance(angle, bool):
            raise TypeError("angle must be numeric (int or float)")
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
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
        if not isinstance(angle, Real) or isinstance(angle, bool):
            raise TypeError("angle must be numeric (int or float)")
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
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
        if f.is_zero():
            raise ValueError("position and target must be different")
        r = (f.cross(up)).normalized()
        if r.is_zero():
            raise ValueError("up must not be parallel to the view direction")
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
        if f.is_zero() or r.is_zero():
            raise ValueError("Euler angles and up_scene must define a non-degenerate camera basis")
        u = (r.cross(f)).normalized()
        if roll != 0:
            if from_degrees: roll = math.radians(roll)
            cr, sr = math.cos(roll), math.sin(roll)
            r, u = r*cr-u*sr, r*sr+u*cr
        return cls.from_axes(position=position, front=f, right=r, up=u)
    
    @classmethod
    def from_quaternion(cls, q:np.ndarray=None, position:vec3=None):
        if position is None: position = vec3.zero()
        if q is None:
            q = np.array([1, 0, 0, 0], dtype=np.float32)
        w, x, y, z = cls._normalized_quaternion(q)
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
        if not all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                   for value in (fov, aspect, near, far)):
            raise TypeError("fov, aspect, near, and far must be finite numbers")
        if not isinstance(from_degrees, bool):
            raise TypeError("from_degrees must be bool")
        if from_degrees:
            fov = math.radians(fov)
        if not 0 < fov < math.pi:
            raise ValueError("fov must be between 0 and pi radians")
        if aspect <= 0:
            raise ValueError("aspect must be greater than zero")
        if near <= 0 or far <= near:
            raise ValueError("near and far must satisfy 0 < near < far")
        f = 1.0/math.tan(fov/2.0)
        m = np.zeros((4,4), dtype=np.float32)
        m[0,0], m[1,1] = f/aspect, f
        m[2,2],m[2,3] = (far+near)/(near-far), 2*far*near/(near-far)
        m[3,2] = -1
        return cls(m)
    
    @classmethod
    def orthographic(cls, left: float=0, right: float=1280, bottom: float=720, top: float=0, near:float=0.1, far:float=300):
        if not all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                   for value in (left, right, bottom, top, near, far)):
            raise TypeError("orthographic bounds must be finite numbers")
        if left == right or bottom == top or near == far:
            raise ValueError("orthographic bounds must have non-zero ranges")
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[0,3] = 2/(right-left), -((right+left)/(right-left))
        m[1,1], m[1,3] = 2/(top-bottom), -((top+bottom)/(top-bottom))
        m[2,2], m[2,3] = -(2/(far-near)), -((far+near)/(far-near))
        return cls(m)
    
    @classmethod
    def rotation_from_quaternion(cls, q: np.ndarray):
        if q is None:
            q = np.array([1, 0, 0, 0], dtype=np.float32)
        w, x, y, z = cls._normalized_quaternion(q)
        xx, yy, zz = 2*x*x, 2*y*y, 2*z*z
        xy, xz, yz = 2*x*y, 2*x*z, 2*y*z
        wx, wy, wz = 2*w*x, 2*w*y, 2*w*z
        m = np.eye(4, dtype=np.float32)
        m[0,0], m[0,1], m[0,2] = 1-(yy+zz), xy-wz, xz+wy
        m[1,0], m[1,1], m[1,2] = xy+wz, 1-(xx+zz), yz-wx
        m[2,0], m[2,1], m[2,2] = xz-wy, yz+wx, 1-(xx+yy)
        return cls(m)

    @staticmethod
    def _normalized_quaternion(q: np.ndarray):
        if not isinstance(q, np.ndarray):
            raise TypeError("q must be a numpy.ndarray")
        if q.size != 4 or not np.issubdtype(q.dtype, np.number):
            raise ValueError("q must contain four numeric components")
        components = q.astype(np.float64, copy=False).reshape(4)
        if not np.all(np.isfinite(components)):
            raise ValueError("q components must be finite")
        magnitude = np.linalg.norm(components)
        if magnitude < 1e-6:
            raise ValueError("q must not be the zero quaternion")
        return components / magnitude
    
    def __matmul__(self, other):
        if isinstance(other, mat4):
            return mat4(self.data @ other.data)
        elif isinstance(other, vec3):
            return self.transform_point(other)
        elif isinstance(other, np.ndarray):
            return self.data @ other
        elif isinstance(other, Real) and not isinstance(other, bool):
            return mat4(self.data*other)
        else:
            raise TypeError("Unsupported type for mat4 multiplication")

    def transform_homogeneous(self, point: vec3, w: float = 1.0) -> np.ndarray:
        if not isinstance(point, vec3):
            raise TypeError("point must be a vec3")
        if not isinstance(w, Real) or not math.isfinite(w):
            raise TypeError("w must be a finite number")
        return self.data @ point.to_homogeneous(w=float(w))

    def transform_point(self, point: vec3) -> vec3:
        result = self.transform_homogeneous(point)
        if abs(result[3]) < 1e-6:
            raise ValueError("Point transforms to a homogeneous coordinate with w=0")
        return vec3.from_numpy(result[:3] / result[3])

    def transform_direction(self, direction: vec3) -> vec3:
        return vec3.from_numpy(self.transform_homogeneous(direction, w=0.0)[:3])

    def transform_points_batch(self, batch, w: float = 1.0):
        if isinstance(batch, np.ndarray):
            if batch.ndim != 2 or batch.shape[1] != 3:
                raise ValueError("batch must have shape (N, 3)")
            if not np.issubdtype(batch.dtype, np.number):
                raise TypeError("batch elements must be numeric")
            points = batch.astype(np.float32, copy=False)
        elif isinstance(batch, (list, tuple)) and all(isinstance(point, vec3) for point in batch):
            arr = np.array([[v.x, v.y, v.z] for v in batch], dtype=np.float32).reshape(-1, 3)
            points = arr
        else:
            raise TypeError("batch must be ndarray (N,3) or list/tuple of vec3")
        if not isinstance(w, Real) or not math.isfinite(w):
            raise TypeError("w must be a finite number")
        homogeneous = np.column_stack((points, np.full(len(points), w, dtype=np.float32)))
        result = (self.data @ homogeneous.T).T
        if w == 0:
            return result[:, :3]
        if np.any(np.abs(result[:, 3]) < 1e-6):
            raise ValueError("A point transforms to a homogeneous coordinate with w=0")
        return result[:, :3] / result[:, 3, np.newaxis]

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
        if not np.allclose(M[3], (0, 0, 0, 1), atol=1e-6):
            raise ValueError("fast_inverse requires an affine matrix")
        try:
            linear_inverse = np.linalg.inv(M[:3, :3])
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not invertible") from None
        M_inv = np.eye(4, dtype=np.float32)
        M_inv[:3, :3] = linear_inverse
        M_inv[:3, 3] = -linear_inverse @ M[:3, 3]
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
        if not np.allclose(M[3], (0, 0, 0, 1), atol=1e-6):
            raise ValueError("Cannot decompose a projective matrix")

        pos = vec3(M[0,3], M[1,3], M[2,3])

        sx = np.linalg.norm(M[:3,0])
        sy = np.linalg.norm(M[:3,1])
        sz = np.linalg.norm(M[:3,2])
        if min(sx, sy, sz) < 1e-6:
            raise ValueError("Cannot decompose a matrix with a zero scale component")

        R = np.zeros((3,3), dtype=np.float32)
        R[:,0] = M[:3,0] / sx
        R[:,1] = M[:3,1] / sy
        R[:,2] = M[:3,2] / sz
        if np.linalg.det(R) < 0:
            sx = -sx
            R[:, 0] = -R[:, 0]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
            raise ValueError("Cannot decompose a matrix containing shear")
        scale = vec3(sx, sy, sz)

        quat = mat4.quaternion_from_matrix(R)

        return pos, quat, scale
    
    def det(self):
        return float(np.linalg.det(self.data))
    
    def lerp(self, other, t):
        return mat4(self.data * (1-t) + other.data * t)
    
    def __eq__(self, value, tol=1e-6):
        if not isinstance(value, mat4):
            return False
        return np.allclose(self.data, value.data, atol=tol)


Mat4 = mat4