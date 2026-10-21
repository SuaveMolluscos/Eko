from pathlib import Path
import tempfile
import unittest

import numpy as np

from eko.assets import load_obj


class TestObjLoader(unittest.TestCase):

    def test_loads_indexed_and_negative_polygon_faces_as_triangles(self):
        contents = """\
# a quad with position/texture/normal references
v 0 0 0
v 2 0 0
v 2 2 0
v 0 2 0
vt 0 0
vn 0 0 1
f 1/1/1 2/1/1 3/1/1 4/1/1
f -4 -2 -1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quad.obj"
            path.write_text(contents, encoding="utf-8")

            vertices, faces = load_obj(path)

        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(faces.dtype, np.int32)
        np.testing.assert_array_equal(faces, ((0, 1, 2), (0, 2, 3), (0, 2, 3)))

    def test_reports_invalid_indices_and_empty_meshes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.obj"
            path.write_text("v 0 0 0\nf 1 2 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "undefined vertex"):
                load_obj(path)

            path.write_text("v 0 0 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no polygon faces"):
                load_obj(path)

    def test_missing_file_propagates_file_not_found(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_obj(Path(directory) / "missing.obj")


if __name__ == "__main__":
    unittest.main()
