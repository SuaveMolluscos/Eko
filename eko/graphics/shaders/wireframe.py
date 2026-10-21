import numpy as np


def wireframe_edges(vertices, faces):
    """Return unique mesh edges as endpoint pairs, preserving vertex positions."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("faces must contain integer indices")
    if faces.size and (np.any(faces < 0) or int(faces.max()) >= len(vertices)):
        raise ValueError("faces contain an invalid vertex index")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must be finite")
    if not len(faces):
        return np.empty((0, 2, 3), dtype=vertices.dtype)

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    return vertices[edges]
