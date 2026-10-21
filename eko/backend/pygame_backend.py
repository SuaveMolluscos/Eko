import math
from numbers import Real

import numpy as np

from eko.backend.events import EkoEvent, EventType, Key
from eko.backend.irenderer import IRenderer


class PygameBackend(IRenderer):
    """Pygame display backend with relative mouse input and RGB framebuffer blits."""

    def __init__(self, capture_mouse: bool = True):
        if not isinstance(capture_mouse, bool):
            raise TypeError("capture_mouse must be bool")
        self._capture_mouse = capture_mouse
        self._pygame = None
        self._screen = None
        self._width = 0
        self._height = 0

    @property
    def initialized(self):
        return self._screen is not None

    @property
    def size(self):
        self._require_initialized()
        return self._width, self._height

    def initialize(self, width: int, height: int, title: str) -> None:
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height must be a positive integer")
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        if self.initialized:
            raise RuntimeError("PygameBackend is already initialized")

        import pygame

        pygame.display.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        pygame.event.set_grab(self._capture_mouse)
        pygame.mouse.set_visible(not self._capture_mouse)
        self._pygame = pygame
        self._screen = screen
        self._width, self._height = width, height

    def clear(self, color: tuple[int, int, int]) -> None:
        self._require_initialized()
        self._screen.fill(self._validate_color(color))

    def blit_pixel_buffer(self, buffer: np.ndarray) -> None:
        self._require_initialized()
        if not isinstance(buffer, np.ndarray):
            raise TypeError("buffer must be a numpy.ndarray")
        if buffer.shape != (self._height, self._width, 3):
            raise ValueError(
                f"buffer must have shape ({self._height}, {self._width}, 3)"
            )
        if buffer.dtype != np.uint8:
            raise TypeError("buffer must have dtype uint8")

        surface_array = np.ascontiguousarray(np.transpose(buffer, (1, 0, 2)))
        self._pygame.surfarray.blit_array(self._screen, surface_array)

    def draw_line(self, p1, p2, color: tuple[int, int, int]) -> None:
        self._require_initialized()
        start = self._validate_point(p1, "p1")
        end = self._validate_point(p2, "p2")
        self._pygame.draw.line(self._screen, self._validate_color(color), start, end)

    def present(self) -> None:
        self._require_initialized()
        self._pygame.display.flip()

    def poll_events(self) -> list[EkoEvent]:
        self._require_initialized()
        events = []
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                events.append(EkoEvent(EventType.QUIT))
            elif event.type in (self._pygame.KEYDOWN, self._pygame.KEYUP):
                event_type = EventType.KEY_DOWN if event.type == self._pygame.KEYDOWN else EventType.KEY_UP
                events.append(EkoEvent(event_type, key=self._translate_key(event.key)))
            elif event.type == self._pygame.MOUSEMOTION:
                events.append(
                    EkoEvent(
                        EventType.MOUSE_MOVE,
                        mouse_delta=tuple(getattr(event, "rel", (0, 0))),
                    )
                )
            elif event.type == self._pygame.MOUSEWHEEL:
                events.append(EkoEvent(EventType.SCROLL, scroll=float(event.y)))
        return events

    def set_mouse_capture(self, captured: bool) -> None:
        self._require_initialized()
        if not isinstance(captured, bool):
            raise TypeError("captured must be bool")
        self._capture_mouse = captured
        self._pygame.event.set_grab(captured)
        self._pygame.mouse.set_visible(not captured)

    def shutdown(self) -> None:
        if self._pygame is None:
            return
        self._pygame.event.set_grab(False)
        self._pygame.mouse.set_visible(True)
        if self._pygame.display.get_init():
            self._pygame.display.quit()
        self._pygame = None
        self._screen = None
        self._width = self._height = 0

    def _require_initialized(self):
        if not self.initialized:
            raise RuntimeError("PygameBackend.initialize() must be called first")

    @staticmethod
    def _validate_color(color):
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError("color must be a three-component RGB tuple")
        if any(
            not isinstance(channel, Real)
            or isinstance(channel, bool)
            or not math.isfinite(channel)
            or not 0 <= channel <= 255
            or int(channel) != channel
            for channel in color
        ):
            raise ValueError("RGB channels must be integers between 0 and 255")
        return tuple(int(channel) for channel in color)

    @staticmethod
    def _validate_point(point, name):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError(f"{name} must contain two coordinates")
        if any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in point
        ):
            raise ValueError(f"{name} coordinates must be finite numbers")
        return tuple(round(value) for value in point)

    def _translate_key(self, pygame_key):
        key_map = {
            self._pygame.K_w: Key.W,
            self._pygame.K_a: Key.A,
            self._pygame.K_s: Key.S,
            self._pygame.K_d: Key.D,
            self._pygame.K_q: Key.Q,
            self._pygame.K_e: Key.E,
            self._pygame.K_SPACE: Key.SPACE,
            self._pygame.K_ESCAPE: Key.ESC,
            self._pygame.K_UP: Key.UP,
            self._pygame.K_DOWN: Key.DOWN,
            self._pygame.K_LEFT: Key.LEFT,
            self._pygame.K_RIGHT: Key.RIGHT,
            self._pygame.K_LSHIFT: Key.SHIFT,
            self._pygame.K_RSHIFT: Key.SHIFT,
            self._pygame.K_LCTRL: Key.CTRL,
            self._pygame.K_RCTRL: Key.CTRL,
            self._pygame.K_LALT: Key.ALT,
            self._pygame.K_RALT: Key.ALT,
            self._pygame.K_RETURN: Key.ENTER,
            self._pygame.K_TAB: Key.TAB,
            self._pygame.K_BACKSPACE: Key.BACKSPACE,
            self._pygame.K_F1: Key.F1,
            self._pygame.K_F2: Key.F2,
            self._pygame.K_F3: Key.F3,
            self._pygame.K_F4: Key.F4,
            self._pygame.K_F5: Key.F5,
            self._pygame.K_F6: Key.F6,
            self._pygame.K_F7: Key.F7,
            self._pygame.K_F8: Key.F8,
            self._pygame.K_F9: Key.F9,
            self._pygame.K_F10: Key.F10,
            self._pygame.K_F11: Key.F11,
            self._pygame.K_F12: Key.F12,
            self._pygame.K_0: Key.ZERO,
            self._pygame.K_1: Key.ONE,
            self._pygame.K_2: Key.TWO,
            self._pygame.K_3: Key.THREE,
            self._pygame.K_4: Key.FOUR,
            self._pygame.K_5: Key.FIVE,
            self._pygame.K_6: Key.SIX,
            self._pygame.K_7: Key.SEVEN,
            self._pygame.K_8: Key.EIGHT,
            self._pygame.K_9: Key.NINE,
        }
        return key_map.get(pygame_key, Key.UNKNOWN)
