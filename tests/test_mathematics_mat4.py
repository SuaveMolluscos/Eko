import unittest

import numpy as np

from eko.mathematics import Mat4, Vec3

mat4 = Mat4
vec3 = Vec3


class TestMat4(unittest.TestCase):

    def assertVecAlmostEqual(self, actual, expected):
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-6)

    def test_initialization_copies_data_and_validates_shape(self):
        source = np.eye(4, dtype=np.float32)
        matrix = mat4(source)
        source[0, 0] = 5
        self.assertEqual(matrix.data[0, 0], 1)
        with self.assertRaises(TypeError):
            mat4("invalid")
        with self.assertRaises(ValueError):
            mat4(np.eye(3))

    def test_identity_and_matrix_composition(self):
        identity = mat4.identity()
        transform = mat4.translation(2, 0, 0) @ mat4.scale(2, 2, 2)
        self.assertVecAlmostEqual(transform @ vec3(1, 0, 0), vec3(4, 0, 0))
        self.assertEqual(identity @ transform, transform)
        self.assertEqual(transform @ identity, transform)

    def test_translation_and_scale_accept_vec3_or_components(self):
        self.assertVecAlmostEqual(mat4.translation(vec3(1, 2, 3)) @ vec3.zero(), vec3(1, 2, 3))
        self.assertVecAlmostEqual(mat4.scale(vec3(2, 3, 4)) @ vec3.one(), vec3(2, 3, 4))
        self.assertVecAlmostEqual(
            mat4.translation(np.float32(1), np.float32(2), np.float32(3)) @ vec3.zero(),
            vec3(1, 2, 3),
        )
        with self.assertRaises(TypeError):
            mat4.translation(1, 2)
        with self.assertRaises(TypeError):
            mat4.scale("1", 2, 3)

    def test_rotation_z_uses_radians_by_default(self):
        rotated = mat4.rotation_z(np.pi / 2) @ vec3(1, 0, 0)
        self.assertVecAlmostEqual(rotated, vec3(0, 1, 0))
        rotated_degrees = mat4.rotation_z(90, from_degrees=True) @ vec3(1, 0, 0)
        self.assertVecAlmostEqual(rotated_degrees, vec3(0, 1, 0))

    def test_transform_direction_ignores_translation(self):
        transform = mat4.translation(5, 6, 7)
        self.assertVecAlmostEqual(transform.transform_direction(vec3(1, 2, 3)), vec3(1, 2, 3))
        with self.assertRaises(ValueError):
            mat4.perspective() @ vec3.zero()

    def test_perspective_transform_divides_by_homogeneous_w(self):
        projection = mat4.perspective(fov=90, aspect=1, near=1, far=10, from_degrees=True)
        clip = projection.transform_homogeneous(vec3(1, 0, -2))
        self.assertAlmostEqual(clip[3], 2)
        self.assertVecAlmostEqual(projection @ vec3(1, 0, -2), vec3(0.5, 0, 1 / 9))
        np.testing.assert_allclose(
            projection.transform_points_batch(np.array([[1, 0, -2]], dtype=np.float32))[0],
            [0.5, 0, 1 / 9],
            atol=1e-6,
        )

    def test_batch_transform_accepts_vectors_and_empty_input(self):
        transform = mat4.translation(1, 2, 3)
        result = transform.transform_points_batch([vec3.zero(), vec3.one()])
        np.testing.assert_allclose(result, [[1, 2, 3], [2, 3, 4]])
        self.assertEqual(transform.transform_points_batch([]).shape, (0, 3))
        directions = transform.transform_points_batch(np.array([[1, 0, 0]]), w=0)
        np.testing.assert_allclose(directions, [[1, 0, 0]])
        with self.assertRaises(ValueError):
            transform.transform_points_batch(np.ones((3, 2)))
        with self.assertRaises(TypeError):
            transform.transform_points_batch([vec3.zero(), 1])

    def test_perspective_and_orthographic_validation(self):
        with self.assertRaises(ValueError):
            mat4.perspective(fov=0)
        with self.assertRaises(ValueError):
            mat4.perspective(aspect=0)
        with self.assertRaises(ValueError):
            mat4.perspective(near=10, far=1)
        with self.assertRaises(ValueError):
            mat4.rotation_z(float("inf"))
        with self.assertRaises(ValueError):
            mat4.orthographic(left=1, right=1)
        with self.assertRaises(ValueError):
            mat4.orthographic(near=2, far=2)

    def test_orthographic_maps_bounds_to_clip_space(self):
        projection = mat4.orthographic(left=0, right=2, bottom=0, top=2, near=1, far=10)
        self.assertVecAlmostEqual(projection @ vec3(0, 0, -1), vec3(-1, -1, -1))
        self.assertVecAlmostEqual(projection @ vec3(2, 2, -10), vec3(1, 1, 1))

    def test_look_at_maps_camera_forward_to_negative_view_z(self):
        view = mat4.look_at(vec3.zero(), vec3.forward(), vec3.up())
        self.assertVecAlmostEqual(view @ vec3(0, 0, 2), vec3(0, 0, -2))
        self.assertVecAlmostEqual(view @ vec3(0, 0, 0), vec3.zero())

    def test_look_at_rejects_degenerate_vectors(self):
        with self.assertRaises(ValueError):
            mat4.look_at(vec3.zero(), vec3.zero(), vec3.up())
        with self.assertRaises(ValueError):
            mat4.look_at(vec3.zero(), vec3.up(), vec3.up())

    def test_inverse_and_fast_inverse_agree_for_affine_transform(self):
        transform = (
            mat4.translation(1, 2, 3)
            @ mat4.rotation_z(0.7)
            @ mat4.scale(2, 3, 4)
        )
        np.testing.assert_allclose(transform.fast_inverse().data, transform.inverse().data, atol=1e-6)
        np.testing.assert_allclose(
            (transform @ transform.fast_inverse()).data,
            np.eye(4),
            atol=1e-6,
        )

    def test_fast_inverse_rejects_projective_and_singular_matrices(self):
        with self.assertRaises(ValueError):
            mat4.perspective().fast_inverse()
        with self.assertRaises(ValueError):
            mat4.scale(1, 0, 1).fast_inverse()
        with self.assertRaises(ValueError):
            mat4.scale(1, 0, 1).inverse()

    def test_transpose_normal_matrix_and_determinant(self):
        matrix = mat4.scale(2, 3, 4)
        self.assertEqual(matrix.det(), 24)
        self.assertEqual(matrix.transpose(), matrix)
        np.testing.assert_allclose(matrix.normal_matrix().data, np.diag([0.5, 1 / 3, 0.25, 1]))

    def test_lerp_and_equality(self):
        start = mat4.identity()
        end = mat4.translation(2, 0, 0)
        midpoint = start.lerp(end, 0.5)
        self.assertVecAlmostEqual(midpoint @ vec3.zero(), vec3(1, 0, 0))
        self.assertEqual(midpoint, mat4.translation(1, 0, 0))
        self.assertNotEqual(midpoint, "not a matrix")

    def test_translation_and_scale_extraction(self):
        transform = mat4.translation(1, 2, 3) @ mat4.scale(2, 3, 4)
        self.assertEqual(transform.get_translation(), vec3(1, 2, 3))
        self.assertEqual(transform.get_scale(), vec3(2, 3, 4))

    def test_quaternion_rotation_uses_right_handed_convention(self):
        half_turn = np.sqrt(0.5)
        quaternion = np.array([half_turn, 0, 0, half_turn], dtype=np.float32)
        rotation = mat4.rotation_from_quaternion(quaternion)
        self.assertVecAlmostEqual(rotation @ vec3(1, 0, 0), vec3(0, 1, 0))
        np.testing.assert_allclose(
            mat4.quaternion_from_matrix(rotation.data[:3, :3]),
            quaternion,
            atol=1e-6,
        )

    def test_camera_quaternion_accepts_flat_components(self):
        half_turn = np.sqrt(0.5)
        quaternion = np.array([half_turn, 0, 0, half_turn], dtype=np.float32)
        view = mat4.from_quaternion(quaternion)
        self.assertVecAlmostEqual(view @ vec3(0, 1, 0), vec3(1, 0, 0))
        self.assertVecAlmostEqual(view @ vec3(0, 0, 1), vec3(0, 0, -1))
        self.assertEqual(mat4.rotation_from_quaternion(quaternion * 3), mat4.rotation_from_quaternion(quaternion))
        with self.assertRaises(ValueError):
            mat4.rotation_from_quaternion(np.zeros(4))
        with self.assertRaises(ValueError):
            mat4.rotation_from_quaternion(np.zeros(3))
        with self.assertRaises(TypeError):
            mat4.rotation_from_quaternion([1, 0, 0, 0])

    def test_decompose_rejects_zero_scale_and_shear(self):
        with self.assertRaises(ValueError):
            mat4.scale(1, 0, 1).decompose()
        shear = mat4.identity()
        shear.data[0, 1] = 0.5
        with self.assertRaises(ValueError):
            shear.decompose()
        with self.assertRaises(ValueError):
            mat4.perspective().decompose()

    def test_decompose_preserves_reflection(self):
        transform = mat4.translation(2, 3, 4) @ mat4.rotation_y(0.4) @ mat4.scale(-2, 3, 4)
        position, quaternion, scale = transform.decompose()
        rebuilt = mat4.translation(position) @ mat4.rotation_from_quaternion(quaternion) @ mat4.scale(scale)
        np.testing.assert_allclose(rebuilt.data, transform.data, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
