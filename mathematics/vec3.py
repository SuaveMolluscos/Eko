import numpy as np

class vec3:
    EPSILON = 1e-6

    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        if not all(isinstance(v, (int, float, np.float32)) for v in (x, y, z)):
            raise TypeError("x, y, z must be int or float")
        self.x = x
        self.y = y
        self.z = z

    def cross(self, other: 'vec3'):
        if isinstance(other, vec3):
            return vec3(self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y - self.y*other.x)
        return NotImplemented

    def magnitude(self):
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5
    
    def normalized(self):
        magnitude = self.magnitude()
        if magnitude < self.EPSILON:
            return vec3.zero()
        return vec3(self.x/magnitude, self.y/magnitude, self.z/magnitude)

    def distance(self, other: 'vec3'):
        if isinstance(other, vec3):
            return (self - other).magnitude()
        return NotImplemented
    
    def lerp(self, other: 'vec3', t: 'float', clamp: bool = False):
        if isinstance(other, vec3):
            try:
                t = float(t)
                if clamp:
                    t = max(0.0, min(1.0, t))
                return self + (other - self) * t
            except (TypeError, ValueError):
                return NotImplemented
        return NotImplemented
    
    def reflect(self, other: 'vec3'):
        if isinstance(other, vec3):
            return self - 2 * (self@other) * other
        return NotImplemented
    
    def clamp(self, min_val: 'vec3', max_val: 'vec3'):
        if isinstance(min_val, vec3):
            min_x, min_y, min_z = min_val.x, min_val.y, min_val.z
        elif isinstance(min_val, (int, float)):
            min_x = min_y = min_z = min_val
        else:
            return NotImplemented
        if isinstance(max_val, vec3):
            max_x, max_y, max_z = max_val.x, max_val.y, max_val.z
        elif isinstance(max_val, (int, float)):
            max_x = max_y = max_z = max_val
        else:
            return NotImplemented
        return vec3(max(min_x, min(max_x, self.x)),
                    max(min_y, min(max_y, self.y)),
                    max(min_z, min(max_z, self.z)))
    
    def angle_to(self, other: 'vec3'):
        if isinstance(other, vec3):
            mag_product = self.magnitude() * other.magnitude()
            if mag_product < self.EPSILON:
                return 0.0
            cos_angle = (self @ other) / mag_product
            cos_angle = max(-1.0, min(1.0, cos_angle))
            return np.arccos(cos_angle)
        return NotImplemented

    def angle_to_degrees(self, other: 'vec3'):
        return np.degrees(self.angle_to(other))
    
    def project_onto(self, other: 'vec3'):
        if isinstance(other, vec3):
            denom = other.x*other.x + other.y*other.y + other.z*other.z
            if denom < self.EPSILON:
                return vec3(0, 0, 0)
            return other * ((self @ other)/denom)
        return NotImplemented

    def is_zero(self) -> bool:
        return -self.EPSILON < self.x < self.EPSILON and -self.EPSILON < self.y < self.EPSILON and -self.EPSILON < self.z < self.EPSILON

        #Array of 3 elements
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
        #homogeneous array
    def to_homogeneous(self, w: float=1.0) -> np.ndarray:
        return np.array([self.x, self.y, self.z, w], dtype=np.float32)

        #Addition
    def __add__(self, other):
        if isinstance(other, vec3):
            return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        try:
            return vec3(self.x + other, self.y + other, self.z + other)
        except TypeError:
            return NotImplemented
    def __radd__(self, other):
        return self.__add__(other)

        #subtraction
    def __sub__(self, other):
        if isinstance(other, vec3):
            return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        try:
            return vec3(self.x - other, self.y - other, self.z - other)
        except TypeError:
            return NotImplemented
    def __rsub__(self, other):
        if isinstance(other, vec3):
            return vec3(other.x - self.x, other.y - self.y, other.z - self.z)
        try:
            return vec3(other - self.x, other - self.y, other - self.z)
        except TypeError:
            return NotImplemented

        #scalar multiplication
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return vec3(self.x * other, self.y * other, self.z * other)
        if isinstance(other, vec3):
            return vec3(self.x*other.x, self.y*other.y, self.z*other.z)
        return NotImplemented
    def __rmul__(self, other):
        return self.__mul__(other)
    
        #Product point
    def __matmul__(self, other):
        if isinstance(other, vec3):
            return self.x*other.x + self.y*other.y + self.z*other.z
        return NotImplemented
    
        #scalar division
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            try:
                return vec3(self.x / other, self.y / other, self.z / other)
            except ZeroDivisionError:
                raise ZeroDivisionError("Cannot divide vec3 by zero")
        return NotImplemented
    def __rtruediv__(self, other):
        return NotImplemented

        #negation
    def __neg__(self):
        return vec3(-self.x, -self.y, -self.z)

        #equal
    def __eq__(self, value):
        if not isinstance(value, vec3):
            return False
        return (abs(self.x - value.x) < self.EPSILON and 
                abs(self.y - value.y) < self.EPSILON and 
                abs(self.z - value.z) < self.EPSILON)
    
        #hasheable
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
        #Get Item
    def __getitem__(self, key):
        if key == 0: return self.x
        if key == 1: return self.y
        if key == 2: return self.z
        if key == -3: return self.x
        if key == -2: return self.y
        if key == -1: return self.z
        raise IndexError("vec3 index out of range")
    
        #Iterador
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

        #console representation
    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

        #static methods
    @staticmethod
    def zero():
        return vec3(0, 0, 0)
    
    @staticmethod
    def one():
        return vec3(1, 1, 1)
    
    @staticmethod
    def up():
        return vec3(0, 1, 0)
    
    @staticmethod
    def down():
        return vec3(0, -1, 0)
    
    @staticmethod
    def right():
        return vec3(1, 0, 0)
    
    @staticmethod
    def left():
        return vec3(-1, 0, 0)
    
    @staticmethod
    def forward():
        return vec3(0, 0, 1)
    
    @staticmethod
    def back():
        return vec3(0, 0, -1)
    
    @staticmethod
    def from_numpy(arr: np.ndarray) -> 'vec3':
        if arr.shape[0] < 3:
            raise ValueError(f"Array must have at least 3 elements, got {arr.shape[0]}")
        return vec3(float(arr[0]), float(arr[1]), float(arr[2]))