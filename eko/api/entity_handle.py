from collections.abc import Iterable
import math
from numbers import Real

from eko.core.components import (
    LightComponent,
    MaterialComponent,
    TransformComponent,
)
from eko.core.world import World
from eko.mathematics.quaternion import Quaternion
from eko.mathematics.vec3 import vec3


class EntityHandle:
    """Fluent, validated access to one entity owned by a World."""

    __slots__ = ("_world", "_entity_id")

    def __init__(self, world: World, entity_id: int):
        if not isinstance(world, World):
            raise TypeError("world must be a World")
        if not isinstance(entity_id, int) or isinstance(entity_id, bool):
            raise TypeError("entity_id must be an integer")
        if not world.is_alive(entity_id):
            raise KeyError(f"Entity {entity_id} does not exist")
        self._world = world
        self._entity_id = entity_id

    @property
    def is_alive(self) -> bool:
        return self._world.is_alive(self._entity_id)

    @property
    def position(self) -> vec3:
        position = self._transform().position
        return vec3(position.x, position.y, position.z)

    @property
    def rotation(self) -> Quaternion:
        current = self._transform().rotation
        return Quaternion(current.w, current.x, current.y, current.z)

    @property
    def scale(self) -> vec3:
        scale = self._transform().scale
        return vec3(scale.x, scale.y, scale.z)

    def translate(self, x, y=None, z=None):
        offset = self._as_vec3(x, y, z, "translation")
        transform = self._transform()
        transform.position = transform.position + offset
        transform.dirty = True
        light = self._world.get_component(self._entity_id, LightComponent)
        if light is not None:
            light.position = vec3(transform.position.x, transform.position.y, transform.position.z)
        return self

    def set_position(self, position, y=None, z=None):
        transform = self._transform()
        transform.position = self._as_vec3(position, y, z, "position")
        transform.dirty = True
        light = self._world.get_component(self._entity_id, LightComponent)
        if light is not None:
            light.position = vec3(transform.position.x, transform.position.y, transform.position.z)
        return self

    def rotate(self, x=0.0, y=0.0, z=0.0):
        angles = (x, y, z)
        self._validate_angles(angles)
        delta = Quaternion.from_euler(yaw=y, pitch=x, roll=z)
        transform = self._transform()
        transform.rotation = (transform.rotation * delta).normalized()
        transform.dirty = True
        return self

    def rotate_y(self, degrees):
        if not isinstance(degrees, Real) or isinstance(degrees, bool):
            raise TypeError("degrees must be a real number")
        return self.rotate(y=degrees * 3.141592653589793 / 180.0)

    def set_rotation(self, rotation: Quaternion):
        if not isinstance(rotation, Quaternion):
            raise TypeError("rotation must be a Quaternion")
        transform = self._transform()
        transform.rotation = rotation.normalized()
        transform.dirty = True
        return self

    def set_scale(self, x, y=None, z=None):
        transform = self._transform()
        transform.scale = self._as_vec3(x, y, z, "scale")
        transform.dirty = True
        return self

    def uniform_scale(self, factor):
        if not isinstance(factor, Real) or isinstance(factor, bool):
            raise TypeError("factor must be a real number")
        return self.set_scale(factor, factor, factor)

    def set_color(self, r, g=None, b=None):
        color = self._as_color(r, g, b)
        material = self._world.get_component(self._entity_id, MaterialComponent)
        if material is None:
            material = MaterialComponent(color=color)
            self._world.add_component(self._entity_id, material)
        else:
            material.color = color
        return self

    def set_material(self, material: MaterialComponent):
        if not isinstance(material, MaterialComponent):
            raise TypeError("material must be a MaterialComponent")
        self._world.add_component(self._entity_id, material)
        return self

    def destroy(self) -> None:
        self._world.destroy_entity(self._entity_id)

    def _transform(self) -> TransformComponent:
        transform = self._world.get_component(self._entity_id, TransformComponent)
        if transform is None:
            raise RuntimeError("entity no longer has a TransformComponent")
        return transform

    @staticmethod
    def _as_vec3(value, y, z, name):
        if isinstance(value, vec3):
            if y is not None or z is not None:
                raise TypeError(f"{name} must be a vec3 or three coordinates")
            if not all(math.isfinite(item) for item in (value.x, value.y, value.z)):
                raise ValueError(f"{name} coordinates must be finite real numbers")
            return vec3(value.x, value.y, value.z)
        if y is None and z is None and isinstance(value, Iterable):
            values = tuple(value)
            if len(values) == 3:
                value, y, z = values
        if y is None or z is None:
            raise TypeError(f"{name} must be a vec3 or three coordinates")
        if not all(
            isinstance(item, Real) and not isinstance(item, bool) and math.isfinite(item)
            for item in (value, y, z)
        ):
            raise ValueError(f"{name} coordinates must be finite real numbers")
        return vec3(value, y, z)

    @staticmethod
    def _validate_angles(angles):
        if not all(
            isinstance(angle, Real) and not isinstance(angle, bool) and math.isfinite(angle)
            for angle in angles
        ):
            raise ValueError("rotation angles must be finite real numbers in radians")

    @staticmethod
    def _as_color(r, g, b):
        if g is None and b is None and isinstance(r, Iterable):
            values = tuple(r)
            if len(values) == 3:
                r, g, b = values
        if g is None or b is None:
            raise TypeError("color must contain three channels")
        channels = (r, g, b)
        if any(
            not isinstance(channel, Real)
            or isinstance(channel, bool)
            or not 0 <= channel <= 255
            or int(channel) != channel
            for channel in channels
        ):
            raise ValueError("color channels must be integers from 0 to 255")
        return tuple(int(channel) for channel in channels)
