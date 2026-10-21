import unittest

import numpy as np

from eko.api import EntityHandle, Scene
from eko.core.components import (
    CameraComponent,
    LightComponent,
    LightType,
    MaterialComponent,
    MeshComponent,
)
from eko.mathematics.primitives import cube
from eko.mathematics.quaternion import Quaternion
from eko.mathematics.vec3 import vec3


class TestEntityHandle(unittest.TestCase):

    def test_fluent_transform_material_and_destruction(self):
        scene = Scene()
        handle = scene.create_entity(
            mesh=cube(),
            position=(1, 2, 3),
            color=(237, 33, 14),
        )

        self.assertIsInstance(handle, EntityHandle)
        self.assertEqual(handle.translate(1, 0, -1).position, vec3(2, 2, 2))
        self.assertEqual(handle.rotate_y(-90).rotation, Quaternion.from_euler(yaw=-np.pi / 2))
        self.assertEqual(handle.set_scale(2, 3, 4).uniform_scale(2).scale, vec3(2, 2, 2))
        self.assertEqual(
            scene.world.get_all_components(MaterialComponent)[0].color,
            (237, 33, 14),
        )
        self.assertIsInstance(scene.world.get_all_components(MeshComponent)[0], MeshComponent)

        handle.destroy()

        self.assertFalse(handle.is_alive)
        self.assertFalse(scene.world.is_alive(0))
        with self.assertRaises(KeyError):
            handle.destroy()

    def test_entity_can_be_created_from_vertex_and_face_arrays(self):
        scene = Scene()
        vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32)
        faces = np.array(((0, 1, 2),), dtype=np.int32)
        entity = scene.create_entity(mesh=(vertices, faces))

        mesh = scene.world.get_all_components(MeshComponent)[0]
        self.assertEqual(mesh.vertices.dtype, np.float32)
        self.assertEqual(mesh.faces.dtype, np.int32)
        self.assertEqual(entity.position, vec3.zero())

    def test_scene_camera_and_light_helpers(self):
        scene = Scene(width=320, height=200)
        camera = scene.create_camera(position=(0, 0, -3), fov=np.pi / 3)
        light = scene.add_light(
            type=LightType.POINT,
            color=(200, 180, 160),
            intensity=0.75,
            position=(1, 2, 3),
        )

        camera_component = scene.world.get_all_components(CameraComponent)[0]
        light_component = next(iter(scene.world.get_all_components(LightComponent).values()))
        self.assertEqual(camera.position, vec3(0, 0, -3))
        self.assertEqual((camera_component.width, camera_component.height), (320, 200))
        self.assertEqual(light_component.type, LightType.POINT)
        self.assertEqual(light_component.position, vec3(1, 2, 3))
        light.translate(1, 0, 0)
        self.assertEqual(light_component.position, vec3(2, 2, 3))
        self.assertTrue(light.is_alive)

    def test_scene_rejects_invalid_mesh_without_creating_entity(self):
        scene = Scene()
        with self.assertRaises(TypeError):
            scene.create_entity(mesh=object())
        self.assertEqual(list(scene.world.query()), [])


if __name__ == "__main__":
    unittest.main()
