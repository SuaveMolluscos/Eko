import unittest
from dataclasses import fields, is_dataclass

import numpy as np

from eko import (
    CameraComponent,
    EventBus,
    LightComponent,
    LightType,
    MaterialComponent,
    MeshComponent,
    Quaternion,
    ShadingMode,
    System,
    TransformComponent,
    Vec3,
    World,
)


class Position:
    def __init__(self, x):
        self.x = x


class Velocity:
    def __init__(self, x):
        self.x = x


class Marker:
    pass


class TestWorld(unittest.TestCase):

    def test_entities_are_monotonic_integer_ids(self):
        world = World()
        first = world.create_entity()
        second = world.create_entity()
        self.assertEqual((first, second), (0, 1))
        self.assertTrue(world.is_alive(first))
        self.assertFalse(world.is_alive(True))

    def test_add_get_replace_and_remove_component(self):
        world = World()
        entity = world.create_entity()
        first = world.add_component(entity, Position(2))
        self.assertIs(world.get_component(entity, Position), first)

        replacement = Position(5)
        self.assertIs(world.add_component(entity, replacement), replacement)
        self.assertIs(world.get_component(entity, Position), replacement)
        self.assertIs(world.remove_component(entity, Position), replacement)
        self.assertIsNone(world.get_component(entity, Position))
        self.assertIsNone(world.remove_component(entity, Position))

    def test_query_uses_smallest_store_and_preserves_requested_order(self):
        world = World()
        entities = [world.create_entity() for _ in range(8)]
        for entity in entities:
            world.add_component(entity, Position(entity))
            world.add_component(entity, Velocity(entity * 10))
        world.add_component(entities[2], Marker())
        world.add_component(entities[6], Marker())

        rows = list(world.query(Velocity, Marker, Position))
        self.assertEqual(
            [(entity, velocity.x, marker.__class__, position.x) for entity, velocity, marker, position in rows],
            [
                (entities[2], 20, Marker, 2),
                (entities[6], 60, Marker, 6),
            ],
        )
        self.assertEqual(list(world.query(Position, Position))[0], (entities[0], world.get_component(entities[0], Position), world.get_component(entities[0], Position)))

    def test_empty_queries_return_entities_and_missing_type_returns_no_rows(self):
        world = World()
        entities = [world.create_entity() for _ in range(3)]
        self.assertEqual({row[0] for row in world.query()}, set(entities))
        self.assertEqual(list(world.query(Position)), [])

    def test_get_all_components_is_a_safe_batch_snapshot(self):
        world = World()
        entity = world.create_entity()
        component = Position(1)
        world.add_component(entity, component)
        snapshot = world.get_all_components(Position)
        self.assertEqual(snapshot, {entity: component})
        snapshot.clear()
        self.assertIs(world.get_component(entity, Position), component)

    def test_destroy_removes_all_components_and_invalidates_entity(self):
        world = World()
        entity = world.create_entity()
        world.add_component(entity, Position(1))
        world.add_component(entity, Velocity(2))
        world.destroy_entity(entity)

        self.assertFalse(world.is_alive(entity))
        self.assertIsNone(world.get_component(entity, Position))
        self.assertEqual(world.get_all_components(Position), {})
        with self.assertRaises(KeyError):
            world.destroy_entity(entity)
        with self.assertRaises(KeyError):
            world.add_component(entity, Marker())

    def test_clear_removes_entities_and_components(self):
        world = World()
        entity = world.create_entity()
        world.add_component(entity, Marker())
        world.clear()
        self.assertEqual(list(world.query()), [])
        self.assertEqual(world.get_all_components(Marker), {})
        self.assertEqual(world.create_entity(), 1)

    def test_entity_and_component_type_validation(self):
        world = World()
        entity = world.create_entity()
        with self.assertRaises(TypeError):
            world.add_component(entity, None)
        with self.assertRaises(TypeError):
            world.add_component(True, Marker())
        with self.assertRaises(TypeError):
            world.get_component(entity, "Marker")
        with self.assertRaises(TypeError):
            list(world.query(Position, "Marker"))
        with self.assertRaises(KeyError):
            world.add_component(500, Marker())


class TestComponents(unittest.TestCase):

    def test_components_are_slotted_dataclasses(self):
        component_types = (
            TransformComponent,
            MeshComponent,
            MaterialComponent,
            CameraComponent,
            LightComponent,
        )
        for component_type in component_types:
            with self.subTest(component_type=component_type.__name__):
                self.assertTrue(is_dataclass(component_type))
                component = component_type()
                self.assertFalse(hasattr(component, "__dict__"))

    def test_defaults_are_independent_and_have_expected_data(self):
        first = TransformComponent()
        second = TransformComponent()
        first.position.x = 10
        self.assertEqual(second.position, Vec3.zero())
        self.assertEqual(first.rotation, Quaternion.identity())
        self.assertTrue(first.dirty)

        first_mesh = MeshComponent()
        second_mesh = MeshComponent()
        self.assertEqual(first_mesh.vertices.dtype, np.float32)
        self.assertEqual(first_mesh.faces.dtype, np.int32)
        self.assertIsNot(first_mesh.vertices, second_mesh.vertices)

        self.assertEqual(MaterialComponent().shading_mode, ShadingMode.FLAT)
        self.assertEqual(LightComponent().type, LightType.DIRECTIONAL)
        self.assertEqual(CameraComponent().width, 1280)

    def test_components_only_expose_declared_fields(self):
        self.assertEqual(
            {field.name for field in fields(MaterialComponent)},
            {"color", "shading_mode"},
        )
        with self.assertRaises(AttributeError):
            MaterialComponent().extra = True


class TestSystem(unittest.TestCase):

    def test_system_contract_and_configuration(self):
        class CounterSystem(System):
            def __init__(self, priority=0, enabled=True):
                super().__init__(priority, enabled)
                self.updates = []

            def update(self, world, dt):
                self.updates.append((world, dt))

        world = World()
        system = CounterSystem(priority=10, enabled=False)
        system.update(world, 0.25)
        self.assertEqual(system.priority, 10)
        self.assertFalse(system.enabled)
        self.assertEqual(system.updates, [(world, 0.25)])
        with self.assertRaises(TypeError):
            System()
        with self.assertRaises(TypeError):
            CounterSystem(priority=1.5)


class TestEventBus(unittest.TestCase):

    def test_subscribe_emit_order_and_keyword_payload(self):
        bus = EventBus()
        received = []
        bus.on("move", lambda dx, dy: received.append(("first", dx, dy)))
        bus.on("move", lambda dx, dy: received.append(("second", dx, dy)))

        count = bus.emit("move", dx=1.5, dy=-2.0)
        self.assertEqual(count, 2)
        self.assertEqual(received, [("first", 1.5, -2.0), ("second", 1.5, -2.0)])

    def test_duplicate_registration_and_unsubscribe(self):
        bus = EventBus()
        calls = []

        def listener(value):
            calls.append(value)

        bus.on("event", listener)
        bus.on("event", listener)
        self.assertEqual(bus.emit("event", value=3), 1)
        self.assertEqual(calls, [3])
        self.assertTrue(bus.off("event", listener))
        self.assertFalse(bus.off("event", listener))
        self.assertEqual(bus.emit("event", value=4), 0)

    def test_dispatch_uses_listener_snapshot(self):
        bus = EventBus()
        calls = []

        def second():
            calls.append("second")

        def first():
            calls.append("first")
            bus.off("snapshot", second)

        bus.on("snapshot", first)
        bus.on("snapshot", second)
        bus.emit("snapshot")
        self.assertEqual(calls, ["first", "second"])
        calls.clear()
        bus.emit("snapshot")
        self.assertEqual(calls, ["first"])

    def test_clear_event_and_all_listeners(self):
        bus = EventBus()
        calls = []
        bus.on("one", lambda: calls.append(1))
        bus.on("two", lambda: calls.append(2))
        bus.clear("one")
        bus.emit("one")
        bus.emit("two")
        self.assertEqual(calls, [2])
        bus.clear()
        self.assertEqual(bus.emit("two"), 0)

    def test_validation_and_listener_exceptions_propagate(self):
        bus = EventBus()
        with self.assertRaises(ValueError):
            bus.on("", lambda: None)
        with self.assertRaises(TypeError):
            bus.on("event", None)

        def failing_listener():
            raise RuntimeError("listener failure")

        bus.on("event", failing_listener)
        with self.assertRaisesRegex(RuntimeError, "listener failure"):
            bus.emit("event")


if __name__ == "__main__":
    unittest.main()
