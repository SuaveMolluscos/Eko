from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from eko.mathematics.quaternion import Quaternion
from eko.mathematics.vec3 import vec3


class ShadingMode(Enum):
    FLAT = auto()
    WIREFRAME = auto()
    GOURAUD = auto()


class LightType(Enum):
    AMBIENT = auto()
    DIRECTIONAL = auto()
    POINT = auto()


@dataclass(slots=True)
class TransformComponent:
    position: vec3 = field(default_factory=vec3.zero)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    scale: vec3 = field(default_factory=vec3.one)
    dirty: bool = True
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))


@dataclass(slots=True)
class MeshComponent:
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    aabb: tuple[vec3, vec3] | None = None
    local_aabb: tuple[vec3, vec3] | None = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("vertices must contain finite values")
        if self.faces.size and (np.any(self.faces < 0) or int(self.faces.max()) >= len(self.vertices)):
            raise ValueError("faces contain an invalid vertex index")
        if self.local_aabb is None and len(self.vertices):
            self.local_aabb = (
                vec3.from_numpy(self.vertices.min(axis=0)),
                vec3.from_numpy(self.vertices.max(axis=0)),
            )

    @classmethod
    def from_mesh(cls, mesh):
        from eko.mathematics.primitives import Mesh

        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a mathematics.Mesh")
        return cls(mesh.vertices.copy(), mesh.indices.astype(np.int32))

    @classmethod
    def from_arrays(cls, vertices, faces):
        return cls(vertices, faces)


@dataclass(slots=True)
class MaterialComponent:
    color: tuple[int, int, int] = (255, 255, 255)
    shading_mode: ShadingMode = ShadingMode.FLAT


@dataclass(slots=True)
class CameraComponent:
    fov: float = 1.0471975511965976
    near: float = 0.1
    far: float = 300.0
    width: int = 1280
    height: int = 720


@dataclass(slots=True)
class LightComponent:
    type: LightType = LightType.DIRECTIONAL
    color: tuple[int, int, int] = (255, 255, 255)
    intensity: float = 1.0
    position: vec3 = field(default_factory=vec3.zero)
