import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from mathematics.vec3 import vec3

class TestVec3(unittest.TestCase):

    def setUp(self):
        self.v = vec3(1, 2, 3)
        self.w = vec3(4, 5, 6)
        self.zero = vec3(0, 0, 0)
        self.eps = vec3.EPSILON

    # ------------------------------------------------------------
    # Inicialización y atributos
    # ------------------------------------------------------------
    def test_initialization(self):
        v = vec3(1.5, 2.5, 3.5)
        self.assertEqual(v.x, 1.5)
        self.assertEqual(v.y, 2.5)
        self.assertEqual(v.z, 3.5)

    def test_initialization_invalid_type(self):
        with self.assertRaises(TypeError):
            vec3("1", 2, 3)
        with self.assertRaises(TypeError):
            vec3(1, "2", 3)
        with self.assertRaises(TypeError):
            vec3(1, 2, "3")

    def test_slots(self):
        # Verifica que no se puedan añadir atributos dinámicos
        with self.assertRaises(AttributeError):
            self.v.new_attr = 10

    # ------------------------------------------------------------
    # Operadores aritméticos
    # ------------------------------------------------------------
    def test_add(self):
        r = self.v + self.w
        self.assertEqual(r, vec3(5, 7, 9))
        r = self.v + 5
        self.assertEqual(r, vec3(6, 7, 8))
        r = 5 + self.v
        self.assertEqual(r, vec3(6, 7, 8))

    def test_sub(self):
        r = self.v - self.w
        self.assertEqual(r, vec3(-3, -3, -3))
        r = self.w - self.v
        self.assertEqual(r, vec3(3, 3, 3))
        r = self.v - 5
        self.assertEqual(r, vec3(-4, -3, -2))
        r = 5 - self.v
        self.assertEqual(r, vec3(4, 3, 2))

    def test_mul_scalar(self):
        r = self.v * 2
        self.assertEqual(r, vec3(2, 4, 6))
        r = 3 * self.v
        self.assertEqual(r, vec3(3, 6, 9))

    def test_mul_hadamard(self):
        r = self.v * self.w
        self.assertEqual(r, vec3(4, 10, 18))

    def test_matmul_dot(self):
        dot = self.v @ self.w
        self.assertEqual(dot, 1*4 + 2*5 + 3*6)  # 32

    def test_truediv(self):
        r = self.v / 2
        self.assertEqual(r, vec3(0.5, 1.0, 1.5))
        with self.assertRaises(ZeroDivisionError):
            _ = self.v / 0

    def test_neg(self):
        r = -self.v
        self.assertEqual(r, vec3(-1, -2, -3))

    # ------------------------------------------------------------
    # Comparación y hash
    # ------------------------------------------------------------
    def test_eq(self):
        self.assertTrue(self.v == vec3(1, 2, 3))
        self.assertFalse(self.v == self.w)
        self.assertFalse(self.v == (1, 2, 3))

    def test_hash(self):
        d = {self.v: "value"}
        self.assertEqual(d[vec3(1, 2, 3)], "value")

    # ------------------------------------------------------------
    # Acceso por índice e iteración
    # ------------------------------------------------------------
    def test_getitem(self):
        self.assertEqual(self.v[0], 1)
        self.assertEqual(self.v[1], 2)
        self.assertEqual(self.v[2], 3)
        self.assertEqual(self.v[-3], 1)
        self.assertEqual(self.v[-2], 2)
        self.assertEqual(self.v[-1], 3)
        with self.assertRaises(IndexError):
            _ = self.v[3]
        with self.assertRaises(IndexError):
            _ = self.v[-4]

    def test_iter(self):
        components = list(self.v)
        self.assertEqual(components, [1, 2, 3])

    # ------------------------------------------------------------
    # Métodos geométricos
    # ------------------------------------------------------------
    def test_cross(self):
        a = vec3(1, 0, 0)
        b = vec3(0, 1, 0)
        self.assertEqual(a.cross(b), vec3(0, 0, 1))
        self.assertEqual(b.cross(a), vec3(0, 0, -1))

    def test_magnitude(self):
        self.assertAlmostEqual(self.v.magnitude(), (1+4+9)**0.5)
        self.assertEqual(self.zero.magnitude(), 0.0)

    def test_normalized(self):
        n = self.v.normalized()
        mag = self.v.magnitude()
        self.assertAlmostEqual(n.x, 1/mag)
        self.assertAlmostEqual(n.y, 2/mag)
        self.assertAlmostEqual(n.z, 3/mag)
        # Vector cero -> zero
        self.assertEqual(self.zero.normalized(), vec3.zero())
        # Vector muy pequeño
        tiny = vec3(1e-7, 0, 0)
        self.assertTrue(tiny.normalized().is_zero())

    def test_distance(self):
        self.assertAlmostEqual(self.v.distance(self.w), ((1-4)**2 + (2-5)**2 + (3-6)**2)**0.5)

    def test_lerp(self):
        a = vec3(0, 0, 0)
        b = vec3(10, 10, 10)
        self.assertEqual(a.lerp(b, 0.5), vec3(5, 5, 5))
        self.assertEqual(a.lerp(b, 0), a)
        self.assertEqual(a.lerp(b, 1), b)
        # Sin clamp
        self.assertEqual(a.lerp(b, 1.5), vec3(15, 15, 15))
        # Con clamp
        self.assertEqual(a.lerp(b, 1.5, clamp=True), b)
        self.assertEqual(a.lerp(b, -0.5, clamp=True), a)

    def test_reflect(self):
        v = vec3(1, -1, 0)
        n = vec3(0, 1, 0)
        r = v.reflect(n)
        self.assertEqual(r, vec3(1, 1, 0))
        # Vector perpendicular -> sin cambio
        v = vec3(1, 0, 0)
        n = vec3(0, 1, 0)
        self.assertEqual(v.reflect(n), v)

    def test_clamp(self):
        v = vec3(5, -10, 20)
        minv = vec3(0, 0, 0)
        maxv = vec3(10, 10, 10)
        self.assertEqual(v.clamp(minv, maxv), vec3(5, 0, 10))
        # Con escalares
        self.assertEqual(v.clamp(0, 10), vec3(5, 0, 10))
        # Mixto
        self.assertEqual(v.clamp(0, maxv), vec3(5, 0, 10))
        self.assertEqual(v.clamp(minv, 10), vec3(5, 0, 10))

    def test_angle_to(self):
        a = vec3(1, 0, 0)
        b = vec3(0, 1, 0)
        self.assertAlmostEqual(a.angle_to(b), np.pi/2)
        self.assertAlmostEqual(a.angle_to_degrees(b), 90.0)
        # Vectores paralelos
        self.assertAlmostEqual(a.angle_to(a), 0.0)
        # Vectores opuestos
        self.assertAlmostEqual(a.angle_to(vec3(-1,0,0)), np.pi)
        # Vector cero -> ángulo 0
        self.assertEqual(a.angle_to(self.zero), 0.0)

    def test_project_onto(self):
        a = vec3(3, 4, 0)
        b = vec3(1, 0, 0)
        proj = a.project_onto(b)
        self.assertEqual(proj, vec3(3, 0, 0))
        # Proyección sobre sí mismo
        self.assertEqual(a.project_onto(a), a)
        # Vector nulo
        self.assertEqual(a.project_onto(self.zero), vec3.zero())

    def test_is_zero(self):
        self.assertFalse(self.v.is_zero())
        self.assertTrue(self.zero.is_zero())
        tiny = vec3(1e-7, 0, 0)
        self.assertTrue(tiny.is_zero())
        not_tiny = vec3(1e-5, 0, 0)
        self.assertFalse(not_tiny.is_zero())

    # ------------------------------------------------------------
    # Métodos estáticos
    # ------------------------------------------------------------
    def test_static_zero(self):
        self.assertEqual(vec3.zero(), vec3(0, 0, 0))

    def test_static_one(self):
        self.assertEqual(vec3.one(), vec3(1, 1, 1))

    def test_static_directions(self):
        self.assertEqual(vec3.up(), vec3(0, 1, 0))
        self.assertEqual(vec3.down(), vec3(0, -1, 0))
        self.assertEqual(vec3.right(), vec3(1, 0, 0))
        self.assertEqual(vec3.left(), vec3(-1, 0, 0))
        self.assertEqual(vec3.forward(), vec3(0, 0, 1))
        self.assertEqual(vec3.back(), vec3(0, 0, -1))

    # ------------------------------------------------------------
    # Conversiones a NumPy
    # ------------------------------------------------------------
    def test_to_numpy(self):
        arr = self.v.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_array_equal(arr, np.array([1, 2, 3], dtype=np.float32))

    def test_to_homogeneous(self):
        arr = self.v.to_homogeneous()
        np.testing.assert_array_equal(arr, [1, 2, 3, 1.0])
        arr = self.v.to_homogeneous(w=0.5)
        np.testing.assert_array_equal(arr, [1, 2, 3, 0.5])

    def test_from_numpy(self):
        arr = np.array([10, 20, 30])
        v = vec3.from_numpy(arr)
        self.assertEqual(v, vec3(10, 20, 30))
        # Con forma incorrecta
        with self.assertRaises(ValueError):
            vec3.from_numpy(np.array([1, 2]))
        # Con tipo diferente (float64)
        arr_f64 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        v = vec3.from_numpy(arr_f64)
        self.assertEqual(v, vec3(1.5, 2.5, 3.5))

    # ------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------
    def test_repr(self):
        self.assertEqual(repr(self.v), "(1, 2, 3)")

if __name__ == '__main__':
    unittest.main()