from pathlib import Path

import numpy as np


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an OBJ's positions and triangulated faces as float32/int32 arrays."""
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")

    source = Path(path)
    vertices = []
    faces = []
    with source.open("r", encoding="utf-8-sig") as obj_file:
        for line_number, raw_line in enumerate(obj_file, start=1):
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"{source}:{line_number}: vertex requires three coordinates")
                try:
                    vertex = tuple(float(value) for value in parts[1:4])
                except ValueError as error:
                    raise ValueError(f"{source}:{line_number}: invalid vertex coordinate") from error
                if not np.all(np.isfinite(vertex)):
                    raise ValueError(f"{source}:{line_number}: vertex coordinates must be finite")
                vertices.append(vertex)
            elif parts[0] == "f":
                if len(parts) < 4:
                    raise ValueError(f"{source}:{line_number}: face requires at least three vertices")
                face = []
                for token in parts[1:]:
                    vertex_reference = token.split("/", 1)[0]
                    try:
                        obj_index = int(vertex_reference)
                    except ValueError as error:
                        raise ValueError(f"{source}:{line_number}: invalid face index {token!r}") from error
                    if obj_index == 0:
                        raise ValueError(f"{source}:{line_number}: OBJ indices are never zero")
                    vertex_index = obj_index - 1 if obj_index > 0 else len(vertices) + obj_index
                    if vertex_index < 0 or vertex_index >= len(vertices):
                        raise ValueError(
                            f"{source}:{line_number}: face references an undefined vertex"
                        )
                    face.append(vertex_index)
                for index in range(1, len(face) - 1):
                    faces.append((face[0], face[index], face[index + 1]))

    if not vertices:
        raise ValueError(f"{source}: no vertex positions found")
    if not faces:
        raise ValueError(f"{source}: no polygon faces found")
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError(f"{source}: vertex count exceeds the supported int32 index range")

    vertex_array = np.asarray(vertices, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int32)
    if not np.all(np.isfinite(vertex_array)):
        raise ValueError(f"{source}: vertex coordinates exceed the supported float32 range")
    return vertex_array, face_array
