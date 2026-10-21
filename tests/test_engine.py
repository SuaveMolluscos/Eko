import threading
import time
import unittest

from eko.backend.events import EkoEvent, EventType, Key
from eko.backend.irenderer import IRenderer
from eko.core.system import System
from eko.core.world import World
from eko.engine import Engine
from eko.graphics.systems.input_system import InputSystem


class FakeRenderer(IRenderer):

    def __init__(self, events=(), fail_initialize=False):
        self.events = list(events)
        self.calls = []
        self.initialized_size = None
        self.fail_initialize = fail_initialize

    def initialize(self, width, height, title):
        self.initialized_size = (width, height, title)
        self.calls.append("initialize")
        if self.fail_initialize:
            raise RuntimeError("backend initialization failed")

    def clear(self, color):
        self.calls.append(("clear", color))

    def blit_pixel_buffer(self, buffer):
        self.calls.append(("blit", buffer))

    def draw_line(self, p1, p2, color):
        self.calls.append(("draw_line", p1, p2, color))

    def present(self):
        self.calls.append("present")

    def poll_events(self):
        self.calls.append("poll_events")
        if self.events:
            return [self.events.pop(0)]
        return []

    def shutdown(self):
        self.calls.append("shutdown")


class RecordingSystem(System):

    def __init__(self, name, log, priority=0, callback=None):
        super().__init__(priority=priority)
        self.name = name
        self.log = log
        self.callback = callback

    def update(self, world, dt):
        self.log.append((self.name, world, dt))
        if self.callback is not None:
            self.callback()


class TestEngine(unittest.TestCase):

    def test_runs_priority_order_and_stops_after_quit_with_real_delta(self):
        world = World()
        backend = FakeRenderer((EkoEvent(EventType.QUIT),))
        log = []
        engine = Engine(world, backend, width=320, height=200, title="Unit test")
        engine.add_system(RecordingSystem("late", log, priority=10)).add_system(
            RecordingSystem("early", log, priority=-10)
        )
        backend_events = []
        engine.event_bus.on(
            "backend_event", lambda event: backend_events.append(event.type)
        )

        engine.run()

        self.assertEqual(backend.initialized_size, (320, 200, "Unit test"))
        self.assertEqual([item[0] for item in log], ["early", "late"])
        self.assertTrue(all(item[1] is world and item[2] >= 0 for item in log))
        self.assertEqual(backend_events, [EventType.QUIT])
        self.assertEqual(backend.calls.count("present"), 1)
        self.assertEqual(backend.calls[-1], "shutdown")
        self.assertFalse(engine.running)

    def test_max_frames_and_enabled_gate(self):
        world = World()
        backend = FakeRenderer()
        log = []
        engine = Engine(world, backend, width=20, height=10)
        engine.add_system(RecordingSystem("disabled", log, priority=0))
        engine.systems[0].enabled = False

        engine.run(max_frames=3)

        self.assertEqual(log, [])
        self.assertEqual(backend.calls.count("present"), 3)
        self.assertEqual(backend.calls.count("poll_events"), 3)

    def test_input_bindings_use_engine_event_bus(self):
        backend = FakeRenderer((
            EkoEvent(EventType.KEY_DOWN, key=Key.W),
            EkoEvent(EventType.QUIT),
        ))
        engine = Engine(World(), backend, width=20, height=10)
        actions = []
        engine.event_bus.on("forward", lambda **data: actions.append(data))
        engine.add_system(InputSystem({Key.W: "forward"}))

        engine.run()

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["value"], 1.0)
        self.assertEqual(actions[0]["key"], Key.W)
        self.assertGreaterEqual(actions[0]["dt"], 0)

    def test_pause_resume_and_stop_from_system_callback(self):
        backend = FakeRenderer()
        engine = Engine(World(), backend, width=20, height=10)
        paused = threading.Event()
        updates = []

        class LifecycleSystem(System):

            def update(self, world, dt):
                updates.append((world, dt))
                if len(updates) == 1:
                    engine.pause()
                    paused.set()
                else:
                    engine.stop()

        engine.add_system(LifecycleSystem())
        errors = []

        def run_engine():
            try:
                engine.run()
            except Exception as error:
                errors.append(error)

        runner = threading.Thread(target=run_engine)
        runner.start()
        self.assertTrue(paused.wait(timeout=2))
        time.sleep(0.08)
        self.assertEqual(len(updates), 1)
        engine.resume()
        runner.join(timeout=2)

        self.assertFalse(runner.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(updates), 2)
        self.assertEqual(backend.calls.count("present"), 2)

    def test_pause_interrupts_frame_limiter_and_excludes_pause_from_delta(self):
        backend = FakeRenderer()
        engine = Engine(World(), backend, width=20, height=10, target_fps=1)
        paused = threading.Event()
        updates = []

        class LifecycleSystem(System):

            def update(self, world, dt):
                updates.append((world, dt))
                if len(updates) == 1:
                    engine.pause()
                    paused.set()
                else:
                    engine.stop()

        engine.add_system(LifecycleSystem())
        runner = threading.Thread(target=engine.run)
        runner.start()
        self.assertTrue(paused.wait(timeout=2))
        time.sleep(0.12)
        engine.resume()
        runner.join(timeout=0.5)

        self.assertFalse(runner.is_alive())
        self.assertEqual(len(updates), 2)
        self.assertIs(updates[1][0], engine.world)
        self.assertLess(updates[1][1], 0.1)

    def test_rejects_duplicate_system_and_invalid_frame_limit(self):
        engine = Engine(World(), FakeRenderer())
        system = RecordingSystem("system", [])
        engine.add_system(system)
        with self.assertRaises(ValueError):
            engine.add_system(system)
        with self.assertRaises(ValueError):
            engine.run(max_frames=-1)

    def test_backend_shutdown_is_attempted_when_initialization_fails(self):
        backend = FakeRenderer(fail_initialize=True)
        engine = Engine(World(), backend, width=20, height=10)

        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            engine.run()

        self.assertEqual(backend.calls[-1], "shutdown")
        self.assertFalse(engine.running)
        self.assertFalse(engine._run_lock.locked())


if __name__ == "__main__":
    unittest.main()
