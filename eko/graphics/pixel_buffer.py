import math
from numbers import Real

import numpy as np

from eko.backend.irenderer import IRenderer


class PixelBuffer:
    """Reusable RGB framebuffer with a float32 z-buffer and triangle rasterizer."""

    __slots__ = ("width", "height", "color", "depth")

    def __init__(self, width: int, height: int):
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height must be a positive integer")
        self.width = width
        self.height = height
        self.color = np.zeros((height, width, 3), dtype=np.uint8)
        self.depth = np.full((height, width), np.inf, dtype=np.float32)

    @property
    def pixels(self):
        return self.color

    @property
    def zbuffer(self):
        return self.depth

    def clear(self, color=(0, 0, 0), depth=math.inf):
        rgb = self._color(color)
        if not isinstance(depth, Real) or isinstance(depth, bool) or math.isnan(depth):
            raise ValueError("depth must be a real number and must not be NaN")
        self.color[:] = rgb
        self.depth.fill(depth)

    def write_triangle(self, p0, p1, p2, color, depth, reciprocal_w=None):
        """Rasterize screen-space vertices (x, y[, z]) using scanline coverage."""
        vertices = np.asarray((p0, p1, p2), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] != 3 or vertices.shape[1] not in (2, 3):
            raise ValueError("triangle points must each contain x, y, and optionally z")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("triangle points must be finite")
        if vertices.shape[1] == 2:
            vertices = np.column_stack((vertices, self._depth_values(depth)))
        else:
            vertices = vertices.copy()
            vertices[:, 2] = self._depth_values(depth, vertices[:, 2])
        triangle_colors = None if color is None else self._triangle_colors(color)
        if reciprocal_w is not None:
            reciprocal_w = np.asarray(reciprocal_w, dtype=np.float64)
            if reciprocal_w.shape != (3,) or not np.all(np.isfinite(reciprocal_w)) or np.any(reciprocal_w <= 0):
                raise ValueError("reciprocal_w must contain three positive finite values")

        area = self._edge(vertices[0], vertices[1], vertices[2, 0], vertices[2, 1])
        if abs(area) < 1e-12:
            return 0
        if area < 0:
            vertices[[1, 2]] = vertices[[2, 1]]
            if triangle_colors is not None and triangle_colors.ndim == 2:
                triangle_colors[[1, 2]] = triangle_colors[[2, 1]]
            if reciprocal_w is not None:
                reciprocal_w[[1, 2]] = reciprocal_w[[2, 1]]
            area = -area

        min_y = max(0, math.ceil(float(vertices[:, 1].min() - 0.5)))
        max_y = min(self.height - 1, math.floor(float(vertices[:, 1].max() - 0.5)))
        min_x = max(0, math.ceil(float(vertices[:, 0].min() - 0.5)))
        max_x = min(self.width - 1, math.floor(float(vertices[:, 0].max() - 0.5)))
        if min_x > max_x or min_y > max_y:
            return 0

        x_coordinates = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        written = 0
        for y in range(min_y, max_y + 1):
            y_coordinate = y + 0.5
            weight0 = self._edge(vertices[1], vertices[2], x_coordinates, y_coordinate) / area
            weight1 = self._edge(vertices[2], vertices[0], x_coordinates, y_coordinate) / area
            weight2 = 1.0 - weight0 - weight1
            covered = (weight0 >= -1e-12) & (weight1 >= -1e-12) & (weight2 >= -1e-12)
            if not np.any(covered):
                continue
            depths = weight0 * vertices[0, 2] + weight1 * vertices[1, 2] + weight2 * vertices[2, 2]
            columns = x_coordinates.astype(np.intp)
            nearer = covered & (depths < self.depth[y, min_x:max_x + 1])
            if np.any(nearer):
                selected = columns[nearer]
                if triangle_colors is None:
                    pass
                elif triangle_colors.ndim == 1:
                    self.color[y, selected] = triangle_colors
                else:
                    color_weights = (
                        np.column_stack((weight0[nearer], weight1[nearer], weight2[nearer]))
                    )
                    if reciprocal_w is not None:
                        color_weights *= reciprocal_w[np.newaxis, :]
                        color_weights /= color_weights.sum(axis=1, keepdims=True)
                    interpolated = (
                        color_weights[:, 0, np.newaxis] * triangle_colors[0]
                        + color_weights[:, 1, np.newaxis] * triangle_colors[1]
                        + color_weights[:, 2, np.newaxis] * triangle_colors[2]
                    )
                    self.color[y, selected] = np.clip(interpolated, 0, 255).astype(np.uint8)
                self.depth[y, selected] = depths[nearer]
                written += int(np.count_nonzero(nearer))
        return written

    def write_line(self, p0, p1, color, depth=(0.0, 0.0)):
        points = np.asarray((p0, p1), dtype=np.float64)
        if points.shape not in ((2, 2), (2, 3)) or not np.all(np.isfinite(points)):
            raise ValueError("line points must be two finite (x, y[, z]) points")
        depths = self._depth_values(
            depth,
            points[:, 2] if points.shape[1] == 3 else None,
            count=2,
        )
        rgb = self._color(color)
        delta = points[1, :2] - points[0, :2]
        steps = max(1, int(math.ceil(float(np.max(np.abs(delta))))))
        t = np.linspace(0.0, 1.0, steps + 1)
        xs = np.rint(points[0, 0] + delta[0] * t).astype(np.int64)
        ys = np.rint(points[0, 1] + delta[1] * t).astype(np.int64)
        line_depth = depths[0] + (depths[1] - depths[0]) * t
        visible = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        xs, ys, line_depth = xs[visible], ys[visible], line_depth[visible]
        nearer = line_depth < self.depth[ys, xs]
        self.color[ys[nearer], xs[nearer]] = rgb
        self.depth[ys[nearer], xs[nearer]] = line_depth[nearer]
        return int(np.count_nonzero(nearer))

    def flush(self, renderer: IRenderer):
        if not isinstance(renderer, IRenderer):
            raise TypeError("renderer must implement IRenderer")
        renderer.blit_pixel_buffer(self.color)

    @staticmethod
    def _edge(a, b, x, y):
        return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])

    @staticmethod
    def _color(color):
        values = np.asarray(color)
        if values.shape != (3,) or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError("color must contain three real numeric channels")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
            raise ValueError("color channels must be finite values in [0, 255]")
        return values.astype(np.uint8)

    @classmethod
    def _triangle_colors(cls, color):
        values = np.asarray(color)
        if values.shape == (3, 3):
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError("vertex colors must contain real numeric channels")
            if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
                raise ValueError("vertex color channels must be finite values in [0, 255]")
            return values.astype(np.float32)
        return cls._color(color)

    @staticmethod
    def _depth_values(depth, fallback=None, count=3):
        values = np.asarray(fallback if fallback is not None else depth, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(count, values, dtype=np.float64)
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"depth must be a finite scalar or {count} finite values")
        return values
