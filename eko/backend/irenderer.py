from abc import ABC, abstractmethod

import numpy as np

from eko.backend.events import EkoEvent


class IRenderer(ABC):
    """Contract between engine systems and a concrete rendering backend."""

    @abstractmethod
    def initialize(self, width: int, height: int, title: str) -> None:
        """Create the rendering target and backend resources."""
        raise NotImplementedError

    @abstractmethod
    def clear(self, color: tuple[int, int, int]) -> None:
        """Clear the frame target using an RGB color."""
        raise NotImplementedError

    @abstractmethod
    def blit_pixel_buffer(self, buffer: np.ndarray) -> None:
        """Present an HxWx3 uint8 RGB framebuffer to the backend target."""
        raise NotImplementedError

    @abstractmethod
    def draw_line(self, p1, p2, color: tuple[int, int, int]) -> None:
        """Draw a 2D line, primarily for debug and wireframe rendering."""
        raise NotImplementedError

    @abstractmethod
    def present(self) -> None:
        """Present the completed frame to the display."""
        raise NotImplementedError

    @abstractmethod
    def poll_events(self) -> list[EkoEvent]:
        """Poll and translate native events to backend-independent events."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources owned by this backend."""
        raise NotImplementedError
