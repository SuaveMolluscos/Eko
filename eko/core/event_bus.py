from collections import defaultdict
from collections.abc import Callable


class EventBus:
    """Synchronous named-event dispatcher. Listener exceptions propagate to emit()."""

    __slots__ = ("_listeners",)

    def __init__(self):
        self._listeners: dict[str, list[Callable[..., object]]] = defaultdict(list)

    def on(self, event_name: str, callback: Callable[..., object]) -> None:
        self._validate_event_name(event_name)
        if not callable(callback):
            raise TypeError("callback must be callable")
        if callback not in self._listeners[event_name]:
            self._listeners[event_name].append(callback)

    def off(self, event_name: str, callback: Callable[..., object]) -> bool:
        self._validate_event_name(event_name)
        if not callable(callback):
            raise TypeError("callback must be callable")
        listeners = self._listeners.get(event_name)
        if listeners is None or callback not in listeners:
            return False
        listeners.remove(callback)
        if not listeners:
            del self._listeners[event_name]
        return True

    def emit(self, event_name: str, **kwargs) -> int:
        self._validate_event_name(event_name)
        listeners = tuple(self._listeners.get(event_name, ()))
        for callback in listeners:
            callback(**kwargs)
        return len(listeners)

    def clear(self, event_name: str | None = None) -> None:
        if event_name is None:
            self._listeners.clear()
            return
        self._validate_event_name(event_name)
        self._listeners.pop(event_name, None)

    @staticmethod
    def _validate_event_name(event_name: str) -> None:
        if not isinstance(event_name, str):
            raise TypeError("event_name must be a string")
        if not event_name:
            raise ValueError("event_name must not be empty")
