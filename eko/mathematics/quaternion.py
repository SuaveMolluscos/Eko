import math
from numbers import Real

import numpy as np


class Quaternion:
    """Unit quaternion in (w, x, y, z) order for right-handed rotations."""

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        values = (w, x, y, z)
        if not all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
            raise TypeError("Quaternion components must be real numbers")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Quaternion components must be finite")
        self.w, self.x, self.y, self.z = map(float, values)

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def from_axis_angle(cls, axis, angle, from_degrees=False):
        from eko.mathematics.vec3 import vec3

        if not isinstance(axis, vec3):
            raise TypeError("axis must be a vec3")
        if not isinstance(angle, Real) or isinstance(angle, bool):
            raise TypeError("angle must be a real number")
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
        if not isinstance(from_degrees, bool):
            raise TypeError("from_degrees must be bool")
        if from_degrees:
            angle = math.radians(angle)
        normalized_axis = axis.normalized()
        if normalized_axis.is_zero():
            raise ValueError("axis must not be zero")
        half_angle = angle * 0.5
        sine = math.sin(half_angle)
        return cls(
            math.cos(half_angle),
            normalized_axis.x * sine,
            normalized_axis.y * sine,
            normalized_axis.z * sine,
        )

    @classmethod
    def from_euler(cls, yaw=0.0, pitch=0.0, roll=0.0, from_degrees=False):
        """Create a rotation as yaw(Y) * pitch(X) * roll(Z), in radians by default."""
        values = (yaw, pitch, roll)
        if not all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
            raise TypeError("yaw, pitch, and roll must be real numbers")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("yaw, pitch, and roll must be finite")
        if not isinstance(from_degrees, bool):
            raise TypeError("from_degrees must be bool")
        if from_degrees:
            yaw, pitch, roll = map(math.radians, values)
        yaw_q = cls._axis_rotation(0, 1, 0, yaw)
        pitch_q = cls._axis_rotation(1, 0, 0, pitch)
        roll_q = cls._axis_rotation(0, 0, 1, roll)
        return (yaw_q * pitch_q * roll_q).normalized()

    @classmethod
    def from_array(cls, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")
        if values.size != 4 or not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError("values must contain four numeric components")
        components = values.astype(np.float64, copy=False).reshape(4)
        return cls(*components)

    @classmethod
    def _axis_rotation(cls, x, y, z, angle):
        half_angle = angle * 0.5
        sine = math.sin(half_angle)
        return cls(math.cos(half_angle), x * sine, y * sine, z * sine)

    def to_array(self):
        return np.array((self.w, self.x, self.y, self.z), dtype=np.float32)

    def magnitude(self):
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self):
        magnitude = self.magnitude()
        if magnitude < 1e-12:
            raise ValueError("Cannot normalize a zero quaternion")
        return Quaternion(self.w / magnitude, self.x / magnitude, self.y / magnitude, self.z / magnitude)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        magnitude_squared = self.magnitude() ** 2
        if magnitude_squared < 1e-24:
            raise ValueError("Cannot invert a zero quaternion")
        conjugate = self.conjugate()
        return Quaternion(
            conjugate.w / magnitude_squared,
            conjugate.x / magnitude_squared,
            conjugate.y / magnitude_squared,
            conjugate.z / magnitude_squared,
        )

    def slerp(self, other, t, clamp=False):
        if not isinstance(other, Quaternion):
            raise TypeError("other must be a Quaternion")
        if not isinstance(t, Real) or isinstance(t, bool) or not math.isfinite(t):
            raise TypeError("t must be a finite real number")
        if not isinstance(clamp, bool):
            raise TypeError("clamp must be bool")
        if clamp:
            t = max(0.0, min(1.0, t))

        start = self.normalized()
        end = other.normalized()
        dot = start.w * end.w + start.x * end.x + start.y * end.y + start.z * end.z
        if dot < 0.0:
            end = Quaternion(-end.w, -end.x, -end.y, -end.z)
            dot = -dot
        dot = max(-1.0, min(1.0, dot))

        if dot > 0.9995:
            return Quaternion(
                start.w + t * (end.w - start.w),
                start.x + t * (end.x - start.x),
                start.y + t * (end.y - start.y),
                start.z + t * (end.z - start.z),
            ).normalized()

        angle = math.acos(dot)
        sine = math.sin(angle)
        start_weight = math.sin((1.0 - t) * angle) / sine
        end_weight = math.sin(t * angle) / sine
        return Quaternion(
            start_weight * start.w + end_weight * end.w,
            start_weight * start.x + end_weight * end.x,
            start_weight * start.y + end_weight * end.y,
            start_weight * start.z + end_weight * end.z,
        ).normalized()

    def to_mat4(self):
        from eko.mathematics.mat4 import mat4

        return mat4.rotation_from_quaternion(self.to_array())

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            )
        if isinstance(other, Real) and not isinstance(other, bool):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return False
        return np.allclose(self.to_array(), other.to_array(), atol=1e-6)

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"


quaternion = Quaternion
