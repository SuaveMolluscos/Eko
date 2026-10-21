from collections.abc import Iterable
import math
from numbers import Real

from eko.api.entity_handle import EntityHandle
from eko.core.components import (
    CameraComponent,
    LightComponent,
    LightType,
    MaterialComponent,
    MeshComponent,
    TransformComponent,
)
from eko.core.event_bus import EventBus
from eko.core.world import World
from eko.mathematics.primitives import Mesh
from eko.mathematics.quaternion import Quaternion
from eko.mathematics.vec3 import vec3


class Scene:
    """Convenient scene-building facade over the underlying ECS World."""

    __slots__ = ("world", "event_bus", "width", "height")

    def __init__(
        self,
        world: World | None = None,
        event_bus: EventBus | None = None,
        width: int = 1280,
        height: int = 720,
    ):
        if world is not None and not isinstance(world, World):
            raise TypeError("world must be a World")
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height must be a positive integer")
        self.world = world if world is not None else World()
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.width = width
        self.height = height

    def create_entity(
        self,
        mesh=None,
        position=(0, 0, 0),
        rotation=None,
        scale=(1, 1, 1),
        color=(255, 255, 255),
    ) -> EntityHandle:
        position = self._as_vec3(position, "position")
        rotation = self._as_rotation(rotation)
        scale = self._as_vec3(scale, "scale")
        mesh_component = self._mesh_component(mesh) if mesh is not None else None
        material = (
            MaterialComponent(color=EntityHandle._as_color(*self._color_args(color)))
            if mesh_component is not None
            else None
        )
        entity_id = self.world.create_entity()
        handle = EntityHandle(self.world, entity_id)
        self.world.add_component(entity_id, TransformComponent(position, rotation, scale))
        if mesh_component is not None:
            self.world.add_component(entity_id, mesh_component)
            self.world.add_component(entity_id, material)
        return handle

    def create_camera(
        self,
        fov=1.0471975511965976,
        near=0.1,
        far=300.0,
        position=(0, 0, 0),
        rotation=None,
        width=None,
        height=None,
    ) -> EntityHandle:
        camera_width = self.width if width is None else width
        camera_height = self.height if height is None else height
        if not isinstance(fov, Real) or isinstance(fov, bool) or not math.isfinite(fov) or not 0 < fov < math.pi:
            raise ValueError("fov must be finite and between 0 and pi radians")
        if not isinstance(near, Real) or isinstance(near, bool) or not math.isfinite(near):
            raise ValueError("near must be finite")
        if not isinstance(far, Real) or isinstance(far, bool) or not math.isfinite(far):
            raise ValueError("far must be finite")
        if near <= 0 or far <= near:
            raise ValueError("camera planes must satisfy 0 < near < far")
        if not isinstance(camera_width, int) or isinstance(camera_width, bool) or camera_width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(camera_height, int) or isinstance(camera_height, bool) or camera_height <= 0:
            raise ValueError("height must be a positive integer")
        position = self._as_vec3(position, "position")
        rotation = self._as_rotation(rotation)
        entity_id = self.world.create_entity()
        self.world.add_component(
            entity_id,
            CameraComponent(
                fov=fov,
                near=near,
                far=far,
                width=camera_width,
                height=camera_height,
            ),
        )
        self.world.add_component(
            entity_id,
            TransformComponent(
                position=position,
                rotation=rotation,
            ),
        )
        return EntityHandle(self.world, entity_id)

    def add_light(
        self,
        type=LightType.DIRECTIONAL,
        color=(255, 255, 255),
        intensity=1.0,
        position=(0, 0, 0),
    ) -> EntityHandle:
        if not isinstance(type, LightType):
            raise TypeError("type must be a LightType")
        if (not isinstance(intensity, Real) or isinstance(intensity, bool)
                or not math.isfinite(intensity) or intensity < 0):
            raise ValueError("intensity must be a finite non-negative real number")
        color = EntityHandle._as_color(*self._color_args(color))
        position = self._as_vec3(position, "position")
        entity_id = self.world.create_entity()
        self.world.add_component(
            entity_id,
            LightComponent(
                type=type,
                color=color,
                intensity=intensity,
                position=position,
            ),
        )
        self.world.add_component(entity_id, TransformComponent(position=position))
        return EntityHandle(self.world, entity_id)

    @staticmethod
    def _as_vec3(value, name):
        if isinstance(value, vec3):
            values = (value.x, value.y, value.z)
        else:
            if not isinstance(value, Iterable):
                raise TypeError(f"{name} must be a vec3 or iterable of three coordinates")
            values = tuple(value)
            if len(values) != 3:
                raise ValueError(f"{name} must contain three coordinates")
        return EntityHandle._as_vec3(values, None, None, name)

    @staticmethod
    def _as_rotation(value):
        if value is None:
            return Quaternion.identity()
        if not isinstance(value, Quaternion):
            raise TypeError("rotation must be a Quaternion")
        return value.normalized()

    @staticmethod
    def _color_args(color):
        if not isinstance(color, Iterable):
            raise TypeError("color must be an iterable of three channels")
        values = tuple(color)
        if len(values) != 3:
            raise ValueError("color must contain three channels")
        return values

    @staticmethod
    def _mesh_component(mesh):
        if isinstance(mesh, MeshComponent):
            return MeshComponent(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
            )
        if isinstance(mesh, Mesh):
            return MeshComponent.from_mesh(mesh)
        if isinstance(mesh, tuple) and len(mesh) == 2:
            return MeshComponent(mesh[0], mesh[1])
        raise TypeError("mesh must be a Mesh, MeshComponent, or (vertices, faces) pair")
