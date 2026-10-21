import numpy as np
from numbers import Real

from eko.core.components import MeshComponent, TransformComponent
from eko.core.system import System
from eko.core.world import World
from eko.mathematics.mat4 import mat4


class TransformSystem(System):
    """Recompute only dirty TRS matrices and their attached mesh world bounds."""

    __slots__ = ("last_batch",)

    def __init__(self, priority=-100, enabled=True):
        super().__init__(priority=priority, enabled=enabled)
        self.last_batch = np.empty((0, 4, 4), dtype=np.float32)

    def update(self, world: World, dt: float) -> None:
        if not isinstance(world, World):
            raise TypeError("world must be a World")
        if not isinstance(dt, Real) or isinstance(dt, bool) or not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")

        dirty_records = []
        for entity_id, transform in world.query(TransformComponent):
            if not transform.dirty:
                continue
            matrix = (
                mat4.translation(transform.position)
                @ transform.rotation.to_mat4()
                @ mat4.scale(transform.scale)
            )
            mesh = world.get_component(entity_id, MeshComponent)
            world_bounds = self._world_bounds(matrix.data, mesh) if mesh is not None else None
            dirty_records.append((transform, matrix.data.copy(), mesh, world_bounds))

        if not dirty_records:
            self.last_batch = np.empty((0, 4, 4), dtype=np.float32)
            return

        self.last_batch = np.stack([record[1] for record in dirty_records]).astype(np.float32, copy=False)
        for transform, matrix, mesh, world_bounds in dirty_records:
            transform.matrix = matrix
            if mesh is not None:
                mesh.aabb = world_bounds
            transform.dirty = False

    @staticmethod
    def _world_bounds(matrix, mesh):
        if not len(mesh.vertices):
            return None
        world_vertices = (matrix @ np.column_stack(
            (mesh.vertices, np.ones(len(mesh.vertices), dtype=np.float32))
        ).T).T
        positions = world_vertices[:, :3] / world_vertices[:, 3, np.newaxis]
        from eko.mathematics.vec3 import vec3

        return vec3.from_numpy(positions.min(axis=0)), vec3.from_numpy(positions.max(axis=0))
