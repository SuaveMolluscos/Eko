from collections.abc import Iterable, Mapping
import math
from numbers import Real

from eko.backend.events import EkoEvent, EventType, Key
from eko.core.event_bus import EventBus
from eko.core.system import System
from eko.core.world import World


class InputSystem(System):
    """Map backend-independent input events to configurable named actions."""

    __slots__ = ("event_bus", "_bindings", "_events")

    def __init__(self, event_bus: EventBus | Mapping | None = None, bindings=None, priority=-50, enabled=True):
        super().__init__(priority=priority, enabled=enabled)
        if isinstance(event_bus, Mapping):
            if bindings is not None:
                raise TypeError("bindings were provided twice")
            bindings, event_bus = event_bus, None
        if event_bus is None:
            event_bus = EventBus()
        if not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus")
        self.event_bus = event_bus
        self._bindings: dict[Key, str] = {}
        self._events: list[EkoEvent] = []
        if bindings is not None:
            for key, action in bindings.items():
                self.bind(key, action)

    @property
    def bindings(self):
        return self._bindings.copy()

    def bind(self, key: Key, action: str) -> None:
        if not isinstance(key, Key):
            raise TypeError("key must be a Key")
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        self._bindings[key] = action

    def unbind(self, key: Key) -> bool:
        if not isinstance(key, Key):
            raise TypeError("key must be a Key")
        return self._bindings.pop(key, None) is not None

    def set_events(self, events: Iterable[EkoEvent]) -> None:
        if isinstance(events, (str, bytes)):
            raise TypeError("events must be an iterable of EkoEvent values")
        queued = list(events)
        if not all(isinstance(event, EkoEvent) for event in queued):
            raise TypeError("events must contain only EkoEvent values")
        self._events = queued

    def update(self, world: World, dt: float) -> None:
        if not isinstance(world, World):
            raise TypeError("world must be a World")
        if not isinstance(dt, Real) or isinstance(dt, bool) or not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be a finite non-negative number")
        events, self._events = self._events, []
        for event in events:
            if event.type in (EventType.KEY_DOWN, EventType.KEY_UP) and event.key is not None:
                action = self._bindings.get(event.key)
                if action is not None:
                    value = 1.0 if event.type is EventType.KEY_DOWN else 0.0
                    self.event_bus.emit(action, value=value, dt=float(dt), key=event.key)
            elif event.type is EventType.MOUSE_MOVE:
                self.event_bus.emit("mouse_move", value=event.mouse_delta, dt=float(dt))
            elif event.type is EventType.SCROLL:
                self.event_bus.emit("scroll", value=event.scroll, dt=float(dt))
            elif event.type is EventType.QUIT:
                self.event_bus.emit("quit", value=True, dt=float(dt))
