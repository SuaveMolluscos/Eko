from eko.mathematics.mat4 import mat4
from eko.mathematics.quaternion import Quaternion
from eko.mathematics.vec3 import vec3


class Transform:
    """Cached TRS transform; edits to exposed vector/quaternion components are detected."""

    __slots__ = ("_position", "_rotation", "_scale", "_matrix", "_dirty", "_cached_state")

    def __init__(self, position=None, rotation=None, scale=None):
        self._position = self._copy_vec3(position, vec3.zero(), "position")
        self._rotation = self._copy_quaternion(rotation)
        self._scale = self._copy_vec3(scale, vec3.one(), "scale")
        self._matrix = None
        self._dirty = True
        self._cached_state = None

    @staticmethod
    def _copy_vec3(value, default, name):
        if value is None:
            value = default
        if not isinstance(value, vec3):
            raise TypeError(f"{name} must be a vec3")
        return vec3(value.x, value.y, value.z)

    @staticmethod
    def _copy_quaternion(value):
        if value is None:
            return Quaternion.identity()
        if not isinstance(value, Quaternion):
            raise TypeError("rotation must be a Quaternion")
        return value.normalized()

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        position = self._copy_vec3(value, vec3.zero(), "position")
        if tuple(position) != tuple(self._position):
            self._position = position
            self._dirty = True

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        rotation = self._copy_quaternion(value)
        if (rotation.w, rotation.x, rotation.y, rotation.z) != (
            self._rotation.w, self._rotation.x, self._rotation.y, self._rotation.z
        ):
            self._rotation = rotation
            self._dirty = True

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        scale = self._copy_vec3(value, vec3.one(), "scale")
        if tuple(scale) != tuple(self._scale):
            self._scale = scale
            self._dirty = True

    @property
    def dirty(self):
        return self._dirty or self._state() != self._cached_state

    def mark_dirty(self):
        self._dirty = True

    def _state(self):
        return (
            self._position.x, self._position.y, self._position.z,
            self._rotation.w, self._rotation.x, self._rotation.y, self._rotation.z,
            self._scale.x, self._scale.y, self._scale.z,
        )

    @property
    def matrix(self):
        state = self._state()
        if self._dirty or state != self._cached_state:
            self._matrix = (
                mat4.translation(self._position)
                @ self._rotation.to_mat4()
                @ mat4.scale(self._scale)
            )
            self._matrix.data.setflags(write=False)
            self._cached_state = state
            self._dirty = False
        return self._matrix

    @property
    def matrix4(self):
        return self.matrix


transform = Transform
