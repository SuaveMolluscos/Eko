from collections.abc import Iterator
from typing import Any, TypeVar


Component = TypeVar("Component")
EntityId = int


class World:
    """ECS registry with integer entities and one component dictionary per type."""

    __slots__ = ("_next_entity_id", "_entities", "_store")

    def __init__(self):
        self._next_entity_id = 0
        self._entities: set[EntityId] = set()
        self._store: dict[type, dict[EntityId, Any]] = {}

    def create_entity(self) -> EntityId:
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self._entities.add(entity_id)
        return entity_id

    def destroy_entity(self, entity_id: EntityId) -> None:
        self._require_entity(entity_id)
        self._entities.remove(entity_id)
        for components in self._store.values():
            components.pop(entity_id, None)

    def is_alive(self, entity_id: EntityId) -> bool:
        if not isinstance(entity_id, int) or isinstance(entity_id, bool):
            return False
        return entity_id in self._entities

    def add_component(self, entity_id: EntityId, component: Component) -> Component:
        self._require_entity(entity_id)
        if component is None:
            raise TypeError("component must not be None")
        component_type = type(component)
        self._store.setdefault(component_type, {})[entity_id] = component
        return component

    def remove_component(self, entity_id: EntityId, component_type: type[Component]) -> Component | None:
        self._require_entity(entity_id)
        self._validate_component_type(component_type)
        return self._store.get(component_type, {}).pop(entity_id, None)

    def get_component(self, entity_id: EntityId, component_type: type[Component]) -> Component | None:
        self._validate_entity_id(entity_id)
        self._validate_component_type(component_type)
        if entity_id not in self._entities:
            return None
        return self._store.get(component_type, {}).get(entity_id)

    def get_all_components(self, component_type: type[Component]) -> dict[EntityId, Component]:
        self._validate_component_type(component_type)
        return self._store.get(component_type, {}).copy()

    def query(self, *component_types: type) -> Iterator[tuple[Any, ...]]:
        for component_type in component_types:
            self._validate_component_type(component_type)

        if not component_types:
            for entity_id in self._entities:
                yield (entity_id,)
            return

        primary_type = min(
            component_types,
            key=lambda component_type: len(self._store.get(component_type, {})),
        )
        primary_store = self._store.get(primary_type, {})
        for entity_id in primary_store:
            if all(
                entity_id in self._store.get(component_type, {})
                for component_type in component_types
            ):
                yield (
                    entity_id,
                    *(self._store[component_type][entity_id] for component_type in component_types),
                )

    def clear(self) -> None:
        self._entities.clear()
        self._store.clear()

    @staticmethod
    def _validate_component_type(component_type: type) -> None:
        if not isinstance(component_type, type):
            raise TypeError("component_type must be a type")

    def _require_entity(self, entity_id: EntityId) -> None:
        self._validate_entity_id(entity_id)
        if entity_id not in self._entities:
            raise KeyError(f"Entity {entity_id} does not exist")

    @staticmethod
    def _validate_entity_id(entity_id: EntityId) -> None:
        if not isinstance(entity_id, int) or isinstance(entity_id, bool):
            raise TypeError("entity_id must be an integer")
