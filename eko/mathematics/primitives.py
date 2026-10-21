from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """Indexed triangle mesh with float32 positions/normals and uint32 triangle indices."""

    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray

    def __post_init__(self):
        raw_vertices = np.asarray(self.vertices)
        raw_indices = np.asarray(self.indices)
        raw_normals = np.asarray(self.normals)
        if not np.issubdtype(raw_vertices.dtype, np.number) or np.issubdtype(raw_vertices.dtype, np.complexfloating):
            raise TypeError("vertices must be real numeric values")
        if not np.issubdtype(raw_normals.dtype, np.number) or np.issubdtype(raw_normals.dtype, np.complexfloating):
            raise TypeError("normals must be real numeric values")
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise TypeError("indices must be integers")
        if raw_indices.size and (np.any(raw_indices < 0) or np.any(raw_indices > np.iinfo(np.uint32).max)):
            raise ValueError("indices must fit the uint32 range")
        vertices = raw_vertices.astype(np.float32)
        indices = raw_indices.astype(np.uint32)
        normals = raw_normals.astype(np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError("vertices must have non-empty shape (N, 3)")
        if indices.ndim != 2 or indices.shape[1] != 3 or len(indices) == 0:
            raise ValueError("indices must have non-empty shape (M, 3)")
        if normals.shape != vertices.shape:
            raise ValueError("normals must have the same shape as vertices")
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
            raise ValueError("vertices and normals must be finite")
        if int(indices.max()) >= len(vertices):
            raise ValueError("indices contain a vertex outside the mesh")
        vertices = vertices.copy()
        indices = indices.copy()
        normals = normals.copy()
        vertices.setflags(write=False)
        indices.setflags(write=False)
        normals.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "normals", normals)

    @property
    def faces(self):
        return self.indices


def cube(size=1.0):
    size = _positive_number(size, "size")
    half = size * 0.5
    faces = (
        ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
        ((0, 0, -1), ((1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1))),
        ((1, 0, 0), ((1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1))),
        ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
        ((0, 1, 0), ((-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1))),
        ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    )
    vertices = []
    normals = []
    indices = []
    for normal, corners in faces:
        base = len(vertices)
        vertices.extend(tuple(component * half for component in corner) for corner in corners)
        normals.extend((normal,) * 4)
        indices.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
    return Mesh(vertices, indices, normals)


def sphere(subdivisions=16, radius=1.0):
    subdivisions = _positive_integer(subdivisions, "subdivisions", minimum=3)
    radius = _positive_number(radius, "radius")
    longitude_segments = subdivisions * 2
    vertices = [(0.0, radius, 0.0)]
    normals = [(0.0, 1.0, 0.0)]

    for latitude in range(1, subdivisions):
        phi = math.pi * latitude / subdivisions
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        for longitude in range(longitude_segments):
            theta = 2.0 * math.pi * longitude / longitude_segments
            normal = (sin_phi * math.cos(theta), cos_phi, sin_phi * math.sin(theta))
            normals.append(normal)
            vertices.append(tuple(component * radius for component in normal))

    bottom_index = len(vertices)
    vertices.append((0.0, -radius, 0.0))
    normals.append((0.0, -1.0, 0.0))
    indices = []

    first_ring = 1
    for longitude in range(longitude_segments):
        current = first_ring + longitude
        following = first_ring + (longitude + 1) % longitude_segments
        indices.append((0, following, current))

    for ring in range(subdivisions - 2):
        upper = 1 + ring * longitude_segments
        lower = upper + longitude_segments
        for longitude in range(longitude_segments):
            next_longitude = (longitude + 1) % longitude_segments
            a = upper + longitude
            b = upper + next_longitude
            c = lower + next_longitude
            d = lower + longitude
            indices.extend(((a, b, c), (a, c, d)))

    last_ring = 1 + (subdivisions - 2) * longitude_segments
    for longitude in range(longitude_segments):
        current = last_ring + longitude
        following = last_ring + (longitude + 1) % longitude_segments
        indices.append((bottom_index, current, following))
    return Mesh(vertices, indices, normals)


def plane(width=1.0, height=1.0):
    width = _positive_number(width, "width")
    height = _positive_number(height, "height")
    return grid(width, height, subdivisions=1)


def grid(width=1.0, depth=1.0, subdivisions=10):
    width = _positive_number(width, "width")
    depth = _positive_number(depth, "depth")
    subdivisions = _positive_integer(subdivisions, "subdivisions", minimum=1)
    half_width, half_depth = width * 0.5, depth * 0.5
    vertices = []
    normals = []
    for row in range(subdivisions + 1):
        z = -half_depth + depth * row / subdivisions
        for column in range(subdivisions + 1):
            x = -half_width + width * column / subdivisions
            vertices.append((x, 0.0, z))
            normals.append((0.0, 1.0, 0.0))

    indices = []
    stride = subdivisions + 1
    for row in range(subdivisions):
        for column in range(subdivisions):
            top_left = row * stride + column
            top_right = top_left + 1
            bottom_left = top_left + stride
            bottom_right = bottom_left + 1
            indices.extend(((top_left, bottom_left, bottom_right), (top_left, bottom_right, top_right)))
    return Mesh(vertices, indices, normals)


def _positive_number(value, name):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return float(value)


def _positive_integer(value, name, minimum):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value
