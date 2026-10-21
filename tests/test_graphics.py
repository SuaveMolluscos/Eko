import unittest

import numpy as np

from eko.backend.events import EkoEvent, EventType, Key
from eko.backend.irenderer import IRenderer
from eko.core.components import (
    CameraComponent,
    MaterialComponent,
    MeshComponent,
    ShadingMode,
    TransformComponent,
)
from eko.core.event_bus import EventBus
from eko.core.world import World
from eko.graphics.culler import backface_cull, frustum_cull
from eko.graphics.pixel_buffer import PixelBuffer
from eko.graphics.shaders.flat import flat_shade
from eko.graphics.shaders.wireframe import wireframe_edges
from eko.graphics.systems.input_system import InputSystem
from eko.graphics.systems.render_system import RenderSystem
from eko.graphics.systems.transform_system import TransformSystem
from eko.mathematics.frustum import Frustum
from eko.mathematics.mat4 import mat4
from eko.mathematics.primitives import cube, plane
from eko.mathematics.quaternion import Quaternion
from eko.mathematics.vec3 import vec3


class FakeRenderer(IRenderer):

    def __init__(self):
        self.buffer = None
        self.presented = False
        self.initialization = None
        self.clear_color = None
        self.lines = []

    def initialize(self, width, height, title):
        self.initialization = (width, height, title)

    def clear(self, color):
        self.clear_color = color

    def blit_pixel_buffer(self, buffer):
        self.buffer = buffer

    def draw_line(self, p1, p2, color):
        self.lines.append((p1, p2, color))

    def present(self):
        self.presented = True

    def poll_events(self):
        return []

    def shutdown(self):
        pass


class TestPixelBuffer(unittest.TestCase):

    def test_clear_reuses_color_and_depth_allocations(self):
        buffer = PixelBuffer(8, 6)
        color_id, depth_id = id(buffer.color), id(buffer.depth)
        buffer.write_triangle((1, 1), (6, 1), (1, 5), (255, 0, 0), 0.5)

        buffer.clear((4, 5, 6))

        self.assertEqual(id(buffer.color), color_id)
        self.assertEqual(id(buffer.depth), depth_id)
        self.assertTrue(np.all(buffer.color == (4, 5, 6)))
        self.assertTrue(np.all(np.isinf(buffer.depth)))

    def test_triangle_rasterization_clips_to_buffer_and_uses_z_buffer(self):
        buffer = PixelBuffer(8, 8)
        first_count = buffer.write_triangle(
            (-2, 1, 0.7), (7, 1, 0.7), (1, 7, 0.7), (255, 0, 0), None
        )
        self.assertGreater(first_count, 0)
        self.assertTrue(np.all(buffer.color[0, 0] == 0))

        buffer.write_triangle((0, 0), (8, 0), (0, 8), (0, 255, 0), 0.2)
        self.assertTrue(np.array_equal(buffer.color[2, 2], (0, 255, 0)))
        farther_count = buffer.write_triangle(
            (0, 0), (8, 0), (0, 8), (0, 0, 255), 0.9
        )
        self.assertEqual(farther_count, 0)
        self.assertTrue(np.array_equal(buffer.color[2, 2], (0, 255, 0)))

    def test_triangle_interpolates_vertex_colors(self):
        buffer = PixelBuffer(8, 8)
        count = buffer.write_triangle(
            (1, 1, 0), (7, 1, 0), (1, 7, 0),
            ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
            None,
        )

        self.assertGreater(count, 0)
        self.assertGreater(len(np.unique(buffer.color.reshape(-1, 3), axis=0)), 3)


class TestGraphicsHelpers(unittest.TestCase):

    def test_frustum_and_backface_culling(self):
        projection = mat4.perspective(np.pi / 2, aspect=1, near=0.1, far=20)
        frustum = Frustum(projection)
        bounds = np.array([
            [[-0.5, -0.5, -3], [0.5, 0.5, -2]],
            [[-0.5, -0.5, 2], [0.5, 0.5, 3]],
            [[50, -0.5, -3], [51, 0.5, -2]],
        ], dtype=np.float32)

        self.assertEqual(frustum_cull(bounds, frustum).tolist(), [True, False, False])
        self.assertEqual(
            backface_cull(
                np.array(((0, 0, -1), (0, 0, 1)), dtype=np.float32),
                np.array((0, 0, 1), dtype=np.float32),
            ).tolist(),
            [True, False],
        )

    def test_flat_shader_and_unique_mesh_edges(self):
        colors = flat_shade(
            np.array(((0, 0, -1), (0, 0, 1)), dtype=np.float32),
            np.array((0, 0, 1), dtype=np.float32),
            (200, 100, 50),
            ambient=0.2,
        )
        np.testing.assert_array_equal(colors, ((200, 100, 50), (40, 20, 10)))

        vertices = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
        edges = wireframe_edges(vertices, np.array(((0, 1, 2), (0, 2, 3))))
        self.assertEqual(edges.shape, (5, 2, 3))


class TestGraphicsSystems(unittest.TestCase):

    def test_transform_system_updates_only_dirty_transforms_and_world_bounds(self):
        world = World()
        entity = world.create_entity()
        transform = TransformComponent(position=vec3(2, 3, 4), scale=vec3(2, 1, 1))
        mesh = MeshComponent.from_mesh(plane(2, 2))
        world.add_component(entity, transform)
        world.add_component(entity, mesh)
        system = TransformSystem()

        system.update(world, 1 / 60)

        self.assertEqual(system.last_batch.shape, (1, 4, 4))
        self.assertFalse(transform.dirty)
        self.assertEqual(mesh.aabb, (vec3(0, 3, 3), vec3(4, 3, 5)))
        previous_matrix = transform.matrix
        system.update(world, 1 / 60)
        self.assertEqual(system.last_batch.shape, (0, 4, 4))
        self.assertIs(transform.matrix, previous_matrix)

    def test_input_system_emits_bound_actions_and_backend_events(self):
        bus = EventBus()
        received = []
        bus.on("move_forward", lambda **payload: received.append(("move", payload)))
        bus.on("mouse_move", lambda **payload: received.append(("mouse", payload)))
        system = InputSystem(bus, {Key.W: "move_forward"})
        system.set_events((
            EkoEvent(EventType.KEY_DOWN, key=Key.W),
            EkoEvent(EventType.MOUSE_MOVE, mouse_delta=(3, -2)),
            EkoEvent(EventType.KEY_UP, key=Key.W),
        ))

        system.update(World(), 0.25)

        self.assertEqual([item[0] for item in received], ["move", "mouse", "move"])
        self.assertEqual(received[0][1]["value"], 1.0)
        self.assertEqual(received[1][1]["value"], (3.0, -2.0))
        self.assertEqual(received[2][1]["value"], 0.0)

    def test_render_system_draws_a_camera_facing_cube(self):
        world = World()
        camera = world.create_entity()
        world.add_component(camera, CameraComponent(width=64, height=64))
        world.add_component(camera, TransformComponent())
        entity = world.create_entity()
        world.add_component(entity, MeshComponent.from_mesh(cube()))
        world.add_component(entity, TransformComponent(position=vec3(0, 0, 3)))
        world.add_component(entity, MaterialComponent(color=(240, 80, 30)))

        pixel_buffer = PixelBuffer(64, 64)
        transform_system = TransformSystem()
        render_system = RenderSystem(pixel_buffer)
        transform_system.update(world, 0.0)
        render_system.update(world, 0.0)

        self.assertEqual(render_system.last_visible_entities, (entity,))
        self.assertTrue(np.any(pixel_buffer.depth < np.inf))
        self.assertTrue(np.any(np.all(pixel_buffer.color != (18, 22, 30), axis=2)))

    def test_render_system_renders_clipped_gouraud_mesh(self):
        world = World()
        camera = world.create_entity()
        world.add_component(camera, CameraComponent(width=48, height=48, near=0.5))
        world.add_component(camera, TransformComponent())
        entity = world.create_entity()
        world.add_component(entity, MeshComponent.from_mesh(cube()))
        world.add_component(
            entity,
            TransformComponent(
                position=vec3(0, 0, 1.0),
                rotation=Quaternion.from_euler(pitch=0.4),
            ),
        )
        world.add_component(
            entity,
            MaterialComponent(color=(200, 100, 50), shading_mode=ShadingMode.GOURAUD),
        )
        pixel_buffer = PixelBuffer(48, 48)
        TransformSystem().update(world, 0.0)
        RenderSystem(pixel_buffer).update(world, 0.0)

        self.assertTrue(np.any(pixel_buffer.depth < np.inf))

    def test_wireframe_uses_depth_prepass_and_renderer_flush(self):
        world = World()
        camera = world.create_entity()
        world.add_component(camera, CameraComponent(width=48, height=48))
        world.add_component(camera, TransformComponent())
        entity = world.create_entity()
        world.add_component(entity, MeshComponent.from_mesh(cube()))
        world.add_component(entity, TransformComponent(position=vec3(0, 0, 3)))
        world.add_component(
            entity,
            MaterialComponent(color=(240, 80, 30), shading_mode=ShadingMode.WIREFRAME),
        )
        pixel_buffer = PixelBuffer(48, 48)
        renderer_system = RenderSystem(pixel_buffer)
        backend = FakeRenderer()

        renderer_system.render(world, backend)

        clear_pixels = np.all(pixel_buffer.color == (18, 22, 30), axis=2)
        self.assertTrue(np.any(np.isfinite(pixel_buffer.depth) & clear_pixels))
        self.assertIs(backend.buffer, pixel_buffer.color)
        self.assertTrue(backend.presented)


if __name__ == "__main__":
    unittest.main()
