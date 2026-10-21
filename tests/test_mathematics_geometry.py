import unittest

import numpy as np

from eko.mathematics import (
    Frustum,
    Mat4,
    Mesh,
    Quaternion,
    Transform,
    Vec3,
    aabb_from_vertices,
    cube,
    face_normal,
    grid,
    plane,
    ray_triangle_intersect,
    sphere,
)

mat4 = Mat4
vec3 = Vec3


class TestQuaternion(unittest.TestCase):

    def test_identity_and_axis_angle_rotation(self):
        self.assertEqual(Quaternion.identity().to_mat4(), mat4.identity())
        rotation = Quaternion.from_axis_angle(vec3.up(), np.pi / 2)
        np.testing.assert_allclose(
            (rotation.to_mat4() @ vec3.forward()).to_numpy(),
            [1, 0, 0],
            atol=1e-6,
        )

    def test_euler_composition_matches_matrix_convention(self):
        yaw, pitch, roll = 0.4, -0.3, 0.2
        quaternion_matrix = Quaternion.from_euler(yaw, pitch, roll).to_mat4()
        euler_matrix = mat4.rotation_y(yaw) @ mat4.rotation_x(pitch) @ mat4.rotation_z(roll)
        np.testing.assert_allclose(quaternion_matrix.data, euler_matrix.data, atol=1e-6)

    def test_from_euler_supports_degrees_and_rejects_invalid_values(self):
        degrees = Quaternion.from_euler(yaw=90, from_degrees=True)
        radians = Quaternion.from_euler(yaw=np.pi / 2)
        self.assertEqual(degrees, radians)
        with self.assertRaises(ValueError):
            Quaternion.from_euler(yaw=float("inf"))
        with self.assertRaises(TypeError):
            Quaternion.from_euler(yaw=True)

    def test_slerp_interpolates_shortest_arc(self):
        start = Quaternion.identity()
        end = Quaternion.from_axis_angle(vec3.up(), np.pi)
        halfway = start.slerp(end, 0.5)
        np.testing.assert_allclose(
            (halfway.to_mat4() @ vec3.forward()).to_numpy(),
            [1, 0, 0],
            atol=1e-6,
        )
        self.assertEqual(start.slerp(-start, 0.5), start)
        self.assertEqual(start.slerp(end, 2, clamp=True), end)

    def test_normalization_inverse_and_invalid_zero(self):
        quaternion = Quaternion(2, 0, 0, 0)
        self.assertEqual(quaternion.normalized(), Quaternion.identity())
        self.assertEqual(quaternion * quaternion.inverse(), Quaternion.identity())
        with self.assertRaises(ValueError):
            Quaternion(0, 0, 0, 0).normalized()
        with self.assertRaises(ValueError):
            Quaternion.from_axis_angle(vec3.zero(), 1)


class TestTransform(unittest.TestCase):

    def test_trs_composition_and_cached_matrix(self):
        transform = Transform(
            position=vec3(1, 2, 3),
            rotation=Quaternion.from_axis_angle(vec3.up(), np.pi / 2),
            scale=vec3(2, 2, 2),
        )
        matrix = transform.matrix
        self.assertIs(transform.matrix, matrix)
        np.testing.assert_allclose((matrix @ vec3.forward()).to_numpy(), [3, 2, 3], atol=1e-6)
        self.assertFalse(transform.dirty)

    def test_assignment_and_in_place_edits_invalidate_cache(self):
        transform = Transform()
        initial = transform.matrix
        transform.position = vec3.zero()
        self.assertIs(transform.matrix, initial)
        transform.position.x = 4
        self.assertTrue(transform.dirty)
        moved = transform.matrix
        self.assertIsNot(moved, initial)
        np.testing.assert_allclose((moved @ vec3.zero()).to_numpy(), [4, 0, 0])
        transform.scale = vec3(3, 3, 3)
        scaled = transform.matrix
        self.assertIsNot(scaled, moved)
        self.assertFalse(transform.dirty)

    def test_inputs_are_copied_and_cached_matrix_is_read_only(self):
        position = vec3(1, 2, 3)
        transform = Transform(position=position)
        position.x = 8
        self.assertEqual(transform.position.x, 1)
        with self.assertRaises(ValueError):
            transform.matrix.data[0, 0] = 4

    def test_invalid_transform_components(self):
        with self.assertRaises(TypeError):
            Transform(position=(0, 0, 0))
        with self.assertRaises(TypeError):
            Transform(rotation=vec3.zero())


class TestFrustum(unittest.TestCase):

    def setUp(self):
        projection = mat4.perspective(fov=90, aspect=1, near=1, far=10, from_degrees=True)
        self.frustum = Frustum(projection)

    def test_extracts_six_normalized_planes(self):
        self.assertEqual(self.frustum.planes.shape, (6, 4))
        np.testing.assert_allclose(np.linalg.norm(self.frustum.planes[:, :3], axis=1), np.ones(6))

    def test_aabb_inside_outside_and_intersecting(self):
        self.assertTrue(self.frustum.test_aabb(vec3(-0.5, -0.5, -3), vec3(0.5, 0.5, -2)))
        self.assertFalse(self.frustum.test_aabb(vec3(20, -1, -4), vec3(21, 1, -3)))
        self.assertFalse(self.frustum.test_aabb(vec3(-0.1, -0.1, -0.5), vec3(0.1, 0.1, -0.2)))
        self.assertTrue(self.frustum.test_aabb(vec3(4, -0.1, -5), vec3(6, 0.1, -4)))

    def test_aabb_accepts_numpy_bounds_and_validates_order(self):
        self.assertTrue(self.frustum.test_aabb(np.array([-0.5, -0.5, -3]), np.array([0.5, 0.5, -2])))
        with self.assertRaises(ValueError):
            self.frustum.test_aabb(vec3(1, 0, 0), vec3(0, 1, 1))
        with self.assertRaises(ValueError):
            Frustum(np.zeros((4, 4), dtype=np.float32))


class TestGeometry(unittest.TestCase):

    def test_face_normal_orientation_and_degenerate_triangle(self):
        normal = face_normal(vec3.zero(), vec3.right(), vec3.up())
        self.assertEqual(normal, vec3.forward())
        self.assertEqual(face_normal(vec3.zero(), vec3.zero(), vec3.zero()), vec3.zero())

    def test_aabb_from_vertices(self):
        minimum, maximum = aabb_from_vertices(
            [vec3(-1, 2, 0), vec3(3, -2, 5), vec3.zero()]
        )
        self.assertEqual(minimum, vec3(-1, -2, 0))
        self.assertEqual(maximum, vec3(3, 2, 5))
        with self.assertRaises(ValueError):
            aabb_from_vertices([])
        with self.assertRaises(ValueError):
            aabb_from_vertices(np.ones((2, 4)))

    def test_ray_triangle_intersections(self):
        v0, v1, v2 = vec3(-1, -1, -4), vec3(1, -1, -4), vec3(0, 1, -4)
        self.assertAlmostEqual(ray_triangle_intersect(vec3.zero(), vec3(0, 0, -2), v0, v1, v2), 2)
        self.assertIsNone(ray_triangle_intersect(vec3(3, 0, 0), vec3(0, 0, -1), v0, v1, v2))
        self.assertIsNone(ray_triangle_intersect(vec3(0, 0, -5), vec3(0, 0, -1), v0, v1, v2))
        self.assertIsNone(ray_triangle_intersect(vec3.zero(), vec3.right(), v0, v1, v2))


class TestPrimitives(unittest.TestCase):

    def assert_mesh_valid(self, mesh):
        self.assertIsInstance(mesh, Mesh)
        self.assertEqual(mesh.vertices.dtype, np.float32)
        self.assertEqual(mesh.normals.dtype, np.float32)
        self.assertEqual(mesh.indices.dtype, np.uint32)
        self.assertEqual(mesh.vertices.shape, mesh.normals.shape)
        self.assertEqual(mesh.indices.ndim, 2)
        self.assertEqual(mesh.indices.shape[1], 3)
        self.assertLess(int(mesh.indices.max()), len(mesh.vertices))
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1, atol=1e-6)
        with self.assertRaises(ValueError):
            mesh.vertices[0, 0] = 5

    def test_cube_has_six_flat_shaded_faces(self):
        mesh = cube()
        self.assert_mesh_valid(mesh)
        self.assertEqual(mesh.vertices.shape, (24, 3))
        self.assertEqual(mesh.indices.shape, (12, 3))
        for triangle in mesh.indices:
            a, b, c = (vec3.from_numpy(mesh.vertices[index]) for index in triangle)
            normal = face_normal(a, b, c)
            self.assertGreater(normal @ vec3.from_numpy(mesh.normals[triangle[0]]), 0.99)

    def test_sphere_subdivisions_and_radius(self):
        mesh = sphere(subdivisions=6, radius=2)
        self.assert_mesh_valid(mesh)
        np.testing.assert_allclose(np.linalg.norm(mesh.vertices, axis=1), 2, atol=1e-6)
        self.assertEqual(mesh.indices.shape, (2 * 6 * (2 * 6 - 2), 3))
        with self.assertRaises(ValueError):
            sphere(subdivisions=2)

    def test_plane_and_subdivided_grid_are_ground_meshes(self):
        flat = plane(width=4, height=2)
        self.assert_mesh_valid(flat)
        self.assertEqual(flat.vertices.shape, (4, 3))
        self.assertEqual(flat.indices.shape, (2, 3))
        np.testing.assert_allclose(flat.normals, np.tile([0, 1, 0], (4, 1)))
        tiled = grid(width=4, depth=2, subdivisions=3)
        self.assert_mesh_valid(tiled)
        self.assertEqual(tiled.vertices.shape, (16, 3))
        self.assertEqual(tiled.indices.shape, (18, 3))
        self.assertEqual(aabb_from_vertices(tiled.vertices), (vec3(-2, 0, -1), vec3(2, 0, 1)))


if __name__ == "__main__":
    unittest.main()
