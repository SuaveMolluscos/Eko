import numpy as np

from eko.mathematics.frustum import Frustum


def frustum_cull(aabbs: np.ndarray, frustum: Frustum) -> np.ndarray:
    """Return a vectorized visible mask for world-space AABBs shaped (N, 2, 3)."""
    if not isinstance(aabbs, np.ndarray):
        raise TypeError("aabbs must be a numpy.ndarray")
    if aabbs.ndim != 3 or aabbs.shape[1:] != (2, 3):
        raise ValueError("aabbs must have shape (N, 2, 3)")
    if not np.issubdtype(aabbs.dtype, np.number) or np.iscomplexobj(aabbs):
        raise TypeError("aabbs must contain real numeric values")
    if not isinstance(frustum, Frustum):
        raise TypeError("frustum must be a Frustum")
    if not np.all(np.isfinite(aabbs)):
        raise ValueError("aabbs must contain finite values")
    if np.any(aabbs[:, 0, :] > aabbs[:, 1, :]):
        raise ValueError("each AABB minimum must be <= its maximum")

    bounds = aabbs.astype(np.float32, copy=False)
    visible = np.ones(len(bounds), dtype=bool)
    for plane in frustum.planes:
        positive = np.where(plane[:3] >= 0, bounds[:, 1, :], bounds[:, 0, :])
        visible &= (positive @ plane[:3] + plane[3]) >= 0
    return visible


def backface_cull(normals: np.ndarray, view_dir: np.ndarray) -> np.ndarray:
    """Return a mask for faces whose normals point toward the viewer.

    ``view_dir`` points from the camera into the scene; visible front faces have
    a negative normal dot product under the engine's counter-clockwise winding.
    """
    if not isinstance(normals, np.ndarray) or not isinstance(view_dir, np.ndarray):
        raise TypeError("normals and view_dir must be numpy.ndarray instances")
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (F, 3)")
    if view_dir.shape != (3,):
        raise ValueError("view_dir must have shape (3,)")
    if (not np.issubdtype(normals.dtype, np.number) or np.iscomplexobj(normals)
            or not np.issubdtype(view_dir.dtype, np.number) or np.iscomplexobj(view_dir)):
        raise TypeError("normals and view_dir must contain real numeric values")
    if not np.all(np.isfinite(normals)) or not np.all(np.isfinite(view_dir)):
        raise ValueError("normals and view_dir must contain finite values")
    return (normals @ view_dir) < 0
