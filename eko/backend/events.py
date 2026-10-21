from dataclasses import dataclass
from enum import Enum, auto
import math
from numbers import Real


class EventType(Enum):
    QUIT = auto()
    KEY_DOWN = auto()
    KEY_UP = auto()
    MOUSE_MOVE = auto()
    SCROLL = auto()


class Key(Enum):
    UNKNOWN = auto()
    W = auto()
    A = auto()
    S = auto()
    D = auto()
    Q = auto()
    E = auto()
    SPACE = auto()
    ESC = auto()
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    SHIFT = auto()
    CTRL = auto()
    ALT = auto()
    ENTER = auto()
    TAB = auto()
    BACKSPACE = auto()
    F1 = auto()
    F2 = auto()
    F3 = auto()
    F4 = auto()
    F5 = auto()
    F6 = auto()
    F7 = auto()
    F8 = auto()
    F9 = auto()
    F10 = auto()
    F11 = auto()
    F12 = auto()
    ZERO = auto()
    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()
    SIX = auto()
    SEVEN = auto()
    EIGHT = auto()
    NINE = auto()


@dataclass(frozen=True, slots=True)
class EkoEvent:
    """Backend-independent input event; scroll is vertical wheel movement."""

    type: EventType
    key: Key | None = None
    mouse_delta: tuple[float, float] = (0.0, 0.0)
    scroll: float = 0.0

    def __post_init__(self):
        if not isinstance(self.type, EventType):
            raise TypeError("type must be an EventType")
        if self.key is not None and not isinstance(self.key, Key):
            raise TypeError("key must be a Key or None")
        if not isinstance(self.mouse_delta, (tuple, list)) or len(self.mouse_delta) != 2:
            raise ValueError("mouse_delta must contain two values")
        if not all(
            isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
            for value in self.mouse_delta
        ):
            raise ValueError("mouse_delta values must be finite real numbers")
        if not isinstance(self.scroll, Real) or isinstance(self.scroll, bool) or not math.isfinite(self.scroll):
            raise ValueError("scroll must be a finite real number")
        object.__setattr__(self, "mouse_delta", tuple(float(value) for value in self.mouse_delta))
        object.__setattr__(self, "scroll", float(self.scroll))
