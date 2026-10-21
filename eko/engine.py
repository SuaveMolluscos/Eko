import math
import threading
import time
from numbers import Real

from eko.backend.events import EkoEvent, EventType
from eko.backend.irenderer import IRenderer
from eko.core.components import CameraComponent
from eko.core.event_bus import EventBus
from eko.core.system import System
from eko.core.world import World
from eko.graphics.systems.input_system import InputSystem
from eko.graphics.systems.render_system import RenderSystem


class Engine:
    """Own the backend lifecycle and run the world's systems once per frame."""

    __slots__ = (
        "world",
        "backend",
        "event_bus",
        "width",
        "height",
        "title",
        "target_fps",
        "_systems",
        "_stop_requested",
        "_wake_event",
        "_run_lock",
        "_running",
        "_paused",
    )

    def __init__(
        self,
        world: World,
        backend: IRenderer,
        *,
        width: int | None = None,
        height: int | None = None,
        title: str = "Eko",
        target_fps: float | None = None,
        event_bus: EventBus | None = None,
    ):
        if not isinstance(world, World):
            raise TypeError("world must be a World")
        if not isinstance(backend, IRenderer):
            raise TypeError("backend must implement IRenderer")
        if (width is None) != (height is None):
            raise ValueError("width and height must be specified together")
        if width is not None:
            self._validate_dimensions(width, height)
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        if target_fps is not None and (
            not isinstance(target_fps, Real)
            or isinstance(target_fps, bool)
            or not math.isfinite(target_fps)
            or target_fps <= 0
        ):
            raise ValueError("target_fps must be a finite positive number or None")
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus")

        self.world = world
        self.backend = backend
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.width = width
        self.height = height
        self.title = title
        self.target_fps = target_fps
        self._systems: list[System] = []
        self._stop_requested = threading.Event()
        self._wake_event = threading.Event()
        self._run_lock = threading.Lock()
        self._running = False
        self._paused = False

    @property
    def systems(self) -> tuple[System, ...]:
        return tuple(self._systems)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return self._paused

    def add_system(self, system: System):
        if not isinstance(system, System):
            raise TypeError("system must inherit from System")
        if self._running:
            raise RuntimeError("systems cannot be added while the engine is running")
        if system in self._systems:
            raise ValueError("system is already registered")
        if isinstance(system, InputSystem):
            system.event_bus = self.event_bus
        if isinstance(system, RenderSystem) and any(
            isinstance(existing, RenderSystem) for existing in self._systems
        ):
            raise ValueError("Engine supports one RenderSystem")
        self._systems.append(system)
        self._systems.sort(key=lambda registered: registered.priority)
        return self

    def pause(self) -> None:
        self._paused = True
        self._wake_event.set()

    def resume(self) -> None:
        self._paused = False
        self._wake_event.set()

    def stop(self) -> None:
        self._stop_requested.set()
        self._wake_event.set()

    def run(self, max_frames: int | None = None) -> None:
        if max_frames is not None and (
            not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 0
        ):
            raise ValueError("max_frames must be a non-negative integer or None")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Engine.run() is already active")

        initialization_attempted = False
        try:
            self._stop_requested.clear()
            self._wake_event.clear()
            self._paused = False
            self._running = True
            width, height = self._resolve_dimensions()
            initialization_attempted = True
            self.backend.initialize(width, height, self.title)
            previous_time = time.perf_counter()
            frame_count = 0
            while not self._stop_requested.is_set():
                if max_frames is not None and frame_count >= max_frames:
                    break
                if self._paused:
                    self._wait_until_resumed()
                    previous_time = time.perf_counter()
                    continue

                frame_start = time.perf_counter()
                dt = max(0.0, frame_start - previous_time)
                previous_time = frame_start
                self._run_frame(dt)
                frame_count += 1
                if self.target_fps is not None:
                    frame_duration = time.perf_counter() - frame_start
                    remaining = max(0.0, 1.0 / self.target_fps - frame_duration)
                    if remaining:
                        self._wake_event.clear()
                        if self._paused:
                            continue
                        if not self._stop_requested.is_set():
                            self._wake_event.wait(timeout=remaining)
        finally:
            self._running = False
            self._paused = False
            self._wake_event.set()
            try:
                if initialization_attempted:
                    self.backend.shutdown()
            finally:
                self._run_lock.release()

    def _run_frame(self, dt: float) -> None:
        events = self.backend.poll_events()
        if not isinstance(events, list) or not all(isinstance(event, EkoEvent) for event in events):
            raise TypeError("backend.poll_events() must return a list of EkoEvent values")
        for event in events:
            self.event_bus.emit("backend_event", event=event)
        for system in self._systems:
            if isinstance(system, InputSystem):
                system.set_events(events)
            if system.enabled:
                system.update(self.world, dt)

        render_systems = [
            system for system in self._systems
            if isinstance(system, RenderSystem) and system.enabled
        ]
        if render_systems:
            render_systems[0].pixel_buffer.flush(self.backend)
        else:
            self.backend.clear((0, 0, 0))
        self.backend.present()
        if any(event.type is EventType.QUIT for event in events):
            self.stop()

    def _resolve_dimensions(self) -> tuple[int, int]:
        render_system = next(
            (system for system in self._systems if isinstance(system, RenderSystem)),
            None,
        )
        if render_system is not None:
            buffer_width = render_system.pixel_buffer.width
            buffer_height = render_system.pixel_buffer.height
            if self.width is not None and (self.width, self.height) != (
                buffer_width, buffer_height
            ):
                raise ValueError("engine dimensions must match the RenderSystem PixelBuffer")
            return buffer_width, buffer_height

        if self.width is not None:
            return self.width, self.height

        cameras = list(self.world.query(CameraComponent))
        if cameras:
            camera = cameras[0][1]
            self._validate_dimensions(camera.width, camera.height)
            if any((candidate.width, candidate.height) != (camera.width, camera.height)
                   for _, candidate in cameras[1:]):
                raise ValueError("all cameras must use the same dimensions")
            return camera.width, camera.height
        return 1280, 720

    def _wait_until_resumed(self) -> None:
        while self._paused and not self._stop_requested.is_set():
            self._wake_event.clear()
            if self._paused and not self._stop_requested.is_set():
                self._wake_event.wait()

    @staticmethod
    def _validate_dimensions(width, height) -> None:
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
