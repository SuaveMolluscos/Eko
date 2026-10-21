import numpy as np

from eko.mathematics.mat4 import mat4
from eko.mathematics.vec3 import vec3


class Frustum:
    """View frustum for right-handed clip space with depth in [-1, 1]."""

    __slots__ = ("planes",)

    def __init__(self, view_projection):
        if isinstance(view_projection, mat4):
            matrix = view_projection.data
        elif isinstance(view_projection, np.ndarray):
            matrix = view_projection
        else:
            raise TypeError("view_projection must be a mat4 or numpy.ndarray")
        if matrix.shape != (4, 4):
            raise ValueError("view_projection must have shape (4, 4)")
        if (not np.issubdtype(matrix.dtype, np.number)
                or np.issubdtype(matrix.dtype, np.complexfloating)
                or not np.all(np.isfinite(matrix))):
            raise ValueError("view_projection must contain finite numeric values")

        rows = matrix.astype(np.float64, copy=False)
        raw_planes = np.array((
            rows[3] + rows[0],
            rows[3] - rows[0],
            rows[3] + rows[1],
            rows[3] - rows[1],
            rows[3] + rows[2],
            rows[3] - rows[2],
        ))
        lengths = np.linalg.norm(raw_planes[:, :3], axis=1)
        if np.any(lengths < 1e-12):
            raise ValueError("view_projection produces a degenerate frustum plane")
        self.planes = (raw_planes / lengths[:, np.newaxis]).astype(np.float32)

    def test_aabb(self, minimum, maximum):
        minimum = self._as_vec3(minimum, "minimum")
        maximum = self._as_vec3(maximum, "maximum")
        if not all(np.isfinite(value) for value in (*minimum, *maximum)):
            raise ValueError("AABB bounds must be finite")
        if any(low > high for low, high in zip(minimum, maximum)):
            raise ValueError("minimum components must not exceed maximum components")

        for plane in self.planes:
            positive = vec3(
                maximum.x if plane[0] >= 0 else minimum.x,
                maximum.y if plane[1] >= 0 else minimum.y,
                maximum.z if plane[2] >= 0 else minimum.z,
            )
            if plane[:3] @ positive.to_numpy() + plane[3] < 0:
                return False
        return True

    @staticmethod
    def _as_vec3(value, name):
        if isinstance(value, vec3):
            return value
        if (isinstance(value, np.ndarray) and value.shape == (3,)
                and np.issubdtype(value.dtype, np.number)
                and not np.issubdtype(value.dtype, np.complexfloating)):
            return vec3.from_numpy(value)
        raise TypeError(f"{name} must be a vec3 or a numeric array with shape (3,)")

    @classmethod
    def from_view_projection(cls, view_projection):
        return cls(view_projection)


frustum = Frustum
