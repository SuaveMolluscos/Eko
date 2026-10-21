import numpy as np

from eko.mathematics.vec3 import vec3


def face_normal(v0, v1, v2):
    """Return a normalized counter-clockwise triangle normal, or zero for a degenerate face."""
    _require_vec3(v0, "v0")
    _require_vec3(v1, "v1")
    _require_vec3(v2, "v2")
    return (v1 - v0).cross(v2 - v0).normalized()


def aabb_from_vertices(vertices):
    """Return (minimum, maximum) vec3 bounds for an Nx3 array or iterable of vec3."""
    if isinstance(vertices, np.ndarray):
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if len(vertices) == 0:
            raise ValueError("vertices must not be empty")
        if (not np.issubdtype(vertices.dtype, np.number)
                or np.issubdtype(vertices.dtype, np.complexfloating)
                or not np.all(np.isfinite(vertices))):
            raise ValueError("vertices must contain finite numeric values")
        values = vertices.astype(np.float32, copy=False)
    else:
        vertices = list(vertices)
        if not vertices:
            raise ValueError("vertices must not be empty")
        if not all(isinstance(vertex, vec3) for vertex in vertices):
            raise TypeError("vertices must contain vec3 instances")
        values = np.array([tuple(vertex) for vertex in vertices], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("vertices must contain finite numeric values")
    return vec3.from_numpy(values.min(axis=0)), vec3.from_numpy(values.max(axis=0))


def ray_triangle_intersect(origin, direction, v0, v1, v2, epsilon=1e-8):
    """Return ray parameter t for the first hit, or None; direction need not be normalized."""
    for name, value in (("origin", origin), ("direction", direction), ("v0", v0), ("v1", v1), ("v2", v2)):
        _require_vec3(value, name)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and greater than zero")

    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = direction.cross(edge2)
    determinant = edge1 @ cross
    if abs(determinant) < epsilon:
        return None
    inverse_determinant = 1.0 / determinant
    offset = origin - v0
    u = (offset @ cross) * inverse_determinant
    if u < 0.0 or u > 1.0:
        return None
    q = offset.cross(edge1)
    v = (direction @ q) * inverse_determinant
    if v < 0.0 or u + v > 1.0:
        return None
    distance = (edge2 @ q) * inverse_determinant
    return distance if distance >= epsilon else None


def _require_vec3(value, name):
    if not isinstance(value, vec3):
        raise TypeError(f"{name} must be a vec3")
