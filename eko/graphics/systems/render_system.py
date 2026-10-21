import numpy as np
from numbers import Real

from eko.core.components import (
    CameraComponent,
    MaterialComponent,
    MeshComponent,
    ShadingMode,
    TransformComponent,
)
from eko.core.system import System
from eko.core.world import World
from eko.graphics.culler import backface_cull, frustum_cull
from eko.graphics.pixel_buffer import PixelBuffer
from eko.graphics.shaders.flat import flat_shade
from eko.graphics.shaders.wireframe import wireframe_edges
from eko.mathematics.frustum import Frustum
from eko.mathematics.mat4 import mat4
from eko.mathematics.vec3 import vec3


class RenderSystem(System):
    """First software-rendering pipeline: world, view, clip, cull, shade, rasterize."""

    __slots__ = ("pixel_buffer", "clear_color", "light_dir", "last_visible_entities")

    def __init__(
        self,
        pixel_buffer: PixelBuffer,
        clear_color=(18, 22, 30),
        light_dir=(0.3, 0.7, 1.0),
        priority=100,
        enabled=True,
    ):
        super().__init__(priority=priority, enabled=enabled)
        if not isinstance(pixel_buffer, PixelBuffer):
            raise TypeError("pixel_buffer must be a PixelBuffer")
        self.pixel_buffer = pixel_buffer
        self.clear_color = PixelBuffer._color(clear_color)
        light = np.asarray(light_dir, dtype=np.float32)
        if light.shape != (3,) or not np.all(np.isfinite(light)) or np.linalg.norm(light) <= 1e-12:
            raise ValueError("light_dir must be a non-zero finite 3D direction")
        self.light_dir = light / np.linalg.norm(light)
        self.last_visible_entities: tuple[int, ...] = ()

    def update(self, world: World, dt: float) -> None:
        if not isinstance(world, World):
            raise TypeError("world must be a World")
        if not isinstance(dt, Real) or isinstance(dt, bool) or not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        self.pixel_buffer.clear(self.clear_color)

        cameras = list(world.query(CameraComponent, TransformComponent))
        if not cameras:
            self.last_visible_entities = ()
            return
        if len(cameras) > 1:
            raise ValueError("RenderSystem currently supports exactly one camera")
        _, camera, camera_transform = cameras[0]
        if camera.width != self.pixel_buffer.width or camera.height != self.pixel_buffer.height:
            raise ValueError("camera dimensions must match the PixelBuffer dimensions")
        self._validate_camera(camera)

        view = mat4.from_quaternion(
            camera_transform.rotation.to_array(),
            position=camera_transform.position,
        )
        projection = mat4.perspective(
            fov=camera.fov,
            aspect=camera.width / camera.height,
            near=camera.near,
            far=camera.far,
        )
        view_projection = projection @ view
        frustum = Frustum(view_projection)
        camera_forward = camera_transform.rotation.to_mat4().transform_direction(vec3.forward()).to_numpy()

        records = []
        for entity_id, mesh, transform in world.query(MeshComponent, TransformComponent):
            if not len(mesh.vertices) or not len(mesh.faces):
                continue
            model = self._transform_matrix(transform)
            world_vertices = self._transform_vertices(model.data, mesh.vertices)
            bounds = np.stack((world_vertices.min(axis=0), world_vertices.max(axis=0)))
            material = world.get_component(entity_id, MaterialComponent) or MaterialComponent()
            records.append((entity_id, mesh, model, world_vertices, bounds, material))

        if not records:
            self.last_visible_entities = ()
            return

        bounds_batch = np.stack([record[4] for record in records])
        visible_mask = frustum_cull(bounds_batch, frustum)
        self.last_visible_entities = tuple(
            record[0] for record, visible in zip(records, visible_mask) if visible
        )

        for record, visible in zip(records, visible_mask):
            if not visible:
                continue
            _, mesh, model, world_vertices, _, material = record
            self._render_mesh(
                mesh,
                model,
                world_vertices,
                material,
                view_projection.data,
                camera_forward,
            )

    def render(self, world: World, renderer) -> None:
        self.update(world, 0.0)
        self.pixel_buffer.flush(renderer)
        renderer.present()

    @staticmethod
    def _validate_camera(camera):
        if not np.isfinite(camera.fov) or not 0 < camera.fov < np.pi:
            raise ValueError("camera fov must be between 0 and pi radians")
        if not np.isfinite(camera.near) or not np.isfinite(camera.far):
            raise ValueError("camera near and far must be finite")
        if camera.near <= 0 or camera.far <= camera.near:
            raise ValueError("camera planes must satisfy 0 < near < far")
        if (not isinstance(camera.width, int) or isinstance(camera.width, bool)
                or not isinstance(camera.height, int) or isinstance(camera.height, bool)
                or camera.width <= 0 or camera.height <= 0):
            raise ValueError("camera width and height must be positive integers")

    @staticmethod
    def _transform_matrix(transform):
        if transform.dirty:
            return (
                mat4.translation(transform.position)
                @ transform.rotation.to_mat4()
                @ mat4.scale(transform.scale)
            )
        return mat4(transform.matrix)

    @staticmethod
    def _transform_vertices(matrix, vertices):
        homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float32)))
        transformed = (matrix @ homogeneous.T).T
        if np.any(np.abs(transformed[:, 3]) < 1e-8):
            raise ValueError("mesh vertex transformed to homogeneous w=0")
        return transformed[:, :3] / transformed[:, 3, np.newaxis]

    def _render_mesh(self, mesh, model, world_vertices, material, view_projection, camera_forward):
        faces = mesh.faces.astype(np.int64, copy=False)
        triangle_vertices = world_vertices[faces]
        normals = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        lengths = np.linalg.norm(normals, axis=1)
        normals = np.divide(
            normals,
            lengths[:, np.newaxis],
            out=np.zeros_like(normals),
            where=lengths[:, np.newaxis] > 1e-12,
        )
        face_mask = backface_cull(normals, camera_forward)
        face_mask &= lengths > 1e-12
        if not np.any(face_mask):
            return

        homogeneous = np.column_stack((mesh.vertices, np.ones(len(mesh.vertices), dtype=np.float32)))
        mvp = view_projection @ model.data
        clip_vertices = (mvp @ homogeneous.T).T
        if material.shading_mode is ShadingMode.WIREFRAME:
            for face_index in np.flatnonzero(face_mask):
                polygon = self._clip_polygon(clip_vertices[faces[face_index]])
                if len(polygon) < 3:
                    continue
                screen = self._to_screen(polygon)
                for index in range(1, len(screen) - 1):
                    self.pixel_buffer.write_triangle(
                        screen[0], screen[index], screen[index + 1], None, None
                    )

            line_color = PixelBuffer._color(material.color)
            for edge in wireframe_edges(mesh.vertices, faces[face_mask]):
                edge_homogeneous = np.column_stack(
                    (edge, np.ones(2, dtype=np.float32))
                )
                clipped = self._clip_segment((mvp @ edge_homogeneous.T).T)
                if clipped is None:
                    continue
                screen = self._to_screen(clipped)
                screen[:, 2] -= 1e-5
                self.pixel_buffer.write_line(screen[0], screen[1], line_color)
            return

        if material.shading_mode is ShadingMode.GOURAUD:
            vertex_normals = np.zeros_like(world_vertices)
            for corner in range(3):
                np.add.at(vertex_normals, faces[:, corner], normals * lengths[:, np.newaxis])
            vertex_lengths = np.linalg.norm(vertex_normals, axis=1)
            vertex_normals = np.divide(
                vertex_normals,
                vertex_lengths[:, np.newaxis],
                out=np.zeros_like(vertex_normals),
                where=vertex_lengths[:, np.newaxis] > 1e-12,
            )
            vertex_colors = flat_shade(vertex_normals, self.light_dir, material.color)
            face_colors = None
        else:
            face_colors = flat_shade(normals, self.light_dir, material.color)
            vertex_colors = None
        for face_index in np.flatnonzero(face_mask):
            face_clip = clip_vertices[faces[face_index]]
            polygon = self._clip_polygon(face_clip)
            if len(polygon) < 3:
                continue
            screen = self._to_screen(polygon)
            if material.shading_mode is ShadingMode.GOURAUD:
                clipped_colors = self._interpolate_clipped_colors(
                    face_clip, vertex_colors[faces[face_index]], polygon
                )
                clipped_reciprocal_w = np.asarray(
                    [1.0 / vertex[3] for vertex in polygon], dtype=np.float64
                )
            else:
                clipped_colors = None
                clipped_reciprocal_w = None
            for index in range(1, len(screen) - 1):
                self.pixel_buffer.write_triangle(
                    screen[0],
                    screen[index],
                    screen[index + 1],
                    (
                        clipped_colors[[0, index, index + 1]]
                        if clipped_colors is not None
                        else face_colors[face_index]
                    ),
                    None,
                    (
                        clipped_reciprocal_w[[0, index, index + 1]]
                        if clipped_reciprocal_w is not None
                        else None
                    ),
                )

    def _to_screen(self, clip_vertices):
        ndc = np.asarray([vertex[:3] / vertex[3] for vertex in clip_vertices], dtype=np.float64)
        screen = np.empty_like(ndc)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * (self.pixel_buffer.width - 1)
        screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * (self.pixel_buffer.height - 1)
        screen[:, 2] = ndc[:, 2]
        return screen

    @staticmethod
    def _clip_polygon(vertices):
        polygon = [vertex.astype(np.float64, copy=True) for vertex in vertices]
        clip_planes = (
            lambda vertex: vertex[0] + vertex[3],
            lambda vertex: vertex[3] - vertex[0],
            lambda vertex: vertex[1] + vertex[3],
            lambda vertex: vertex[3] - vertex[1],
            lambda vertex: vertex[2] + vertex[3],
            lambda vertex: vertex[3] - vertex[2],
        )
        for distance_to_plane in clip_planes:
            if not polygon:
                break
            clipped = []
            previous = polygon[-1]
            previous_distance = distance_to_plane(previous)
            for current in polygon:
                current_distance = distance_to_plane(current)
                if current_distance >= 0:
                    if previous_distance < 0:
                        amount = previous_distance / (previous_distance - current_distance)
                        clipped.append(previous + amount * (current - previous))
                    clipped.append(current)
                elif previous_distance >= 0:
                    amount = previous_distance / (previous_distance - current_distance)
                    clipped.append(previous + amount * (current - previous))
                previous, previous_distance = current, current_distance
            polygon = clipped
        return polygon

    @staticmethod
    def _clip_segment(vertices):
        start, end = (vertex.astype(np.float64, copy=True) for vertex in vertices)
        clip_planes = (
            lambda vertex: vertex[0] + vertex[3],
            lambda vertex: vertex[3] - vertex[0],
            lambda vertex: vertex[1] + vertex[3],
            lambda vertex: vertex[3] - vertex[1],
            lambda vertex: vertex[2] + vertex[3],
            lambda vertex: vertex[3] - vertex[2],
        )
        for distance_to_plane in clip_planes:
            start_distance = distance_to_plane(start)
            end_distance = distance_to_plane(end)
            if start_distance < 0 and end_distance < 0:
                return None
            if start_distance < 0:
                amount = start_distance / (start_distance - end_distance)
                start = start + amount * (end - start)
            elif end_distance < 0:
                amount = start_distance / (start_distance - end_distance)
                end = start + amount * (end - start)
        if start[3] <= 1e-10 or end[3] <= 1e-10:
            return None
        return np.stack((start, end))

    @classmethod
    def _interpolate_clipped_colors(cls, original_vertices, original_colors, clipped_vertices):
        result = []
        for clipped in clipped_vertices:
            barycentric = np.linalg.lstsq(original_vertices.T, clipped, rcond=None)[0]
            result.append(barycentric @ original_colors)
        return np.asarray(result, dtype=np.float32)
