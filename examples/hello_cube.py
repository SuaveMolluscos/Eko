"""Render a rotating cube with Eko's software-rendering pipeline."""

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eko import (
    EntityHandle,
    Engine,
    InputSystem,
    PixelBuffer,
    RenderSystem,
    Scene,
    System,
    TransformSystem,
    cube,
)
from eko.backend import Key, PygameBackend


class RotateCubeSystem(System):
    """Apply a small frame-rate-independent rotation to the sample cube."""

    def __init__(self, entity: EntityHandle):
        super().__init__(priority=-150)
        self.entity = entity

    def update(self, world, dt: float) -> None:
        _ = world
        if not self.entity.is_alive:
            return
        self.entity.rotate(y=0.8 * dt, x=0.35 * dt)


def main(max_frames: int | None = None) -> None:
    width, height = 800, 600
    scene = Scene(width=width, height=height)
    scene.create_camera()
    box = scene.create_entity(
        mesh=cube(),
        position=(0, 0, 4),
        color=(237, 90, 45),
    )

    engine = Engine(
        scene.world,
        PygameBackend(capture_mouse=False),
        event_bus=scene.event_bus,
        title="Eko - Hello Cube",
        target_fps=60,
    )
    engine.event_bus.on(
        "exit",
        lambda value, **_: engine.stop() if value else None,
    )
    engine.add_system(RotateCubeSystem(box))
    engine.add_system(TransformSystem())
    engine.add_system(InputSystem({Key.ESC: "exit"}))
    engine.add_system(RenderSystem(PixelBuffer(width, height)))
    engine.run(max_frames=max_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="stop after this many frames instead of running until the window closes",
    )
    arguments = parser.parse_args()
    main(max_frames=arguments.frames)
