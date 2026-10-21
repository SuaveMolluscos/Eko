from abc import ABC, abstractmethod

from eko.core.world import World


class System(ABC):
    """Base contract for update systems, ordered by ascending priority."""

    __slots__ = ("priority", "enabled")

    def __init__(self, priority: int = 0, enabled: bool = True):
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self.priority = priority
        self.enabled = enabled

    @abstractmethod
    def update(self, world: World, dt: float) -> None:
        """Update this system using elapsed frame time in seconds."""
        raise NotImplementedError
