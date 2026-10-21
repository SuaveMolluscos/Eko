# Eko Engine

[![Tests](https://github.com/SuaveMolluscos/Eko/actions/workflows/tests.yml/badge.svg)](https://github.com/SuaveMolluscos/Eko/actions/workflows/tests.yml)

> An experimental 3D framework in Python, built from the math up.

Eko is an open-source project for exploring the foundations of 3D graphics and engine architecture. It aims to make the rendering pipeline understandable and extensible instead of hiding it behind a high-level engine.

**Project stage:** mathematical core, a Pygame backend, ECS infrastructure, a software-rendering pipeline, a scene API, OBJ loading, and a managed engine loop.

## Project goals

- Build a modular foundation for experimenting with 3D graphics.
- Keep rendering and mathematical systems understandable and independently testable.
- Explore transforms, camera projection, visibility, and software rendering.
- Use Python to validate architecture and algorithms before considering a native backend.

Eko is a learning and research project, not yet a general-purpose game engine.

## Current features

- `vec3` and `mat4` foundations, including vector operations, transformations, projections, camera matrices, and matrix inversion.
- Quaternion rotations, spherical interpolation, and conversion to 4x4 rotation matrices.
- Cached translation-rotation-scale (`TRS`) transforms.
- View-frustum plane extraction and axis-aligned bounding-box (`AABB`) tests.
- Geometry helpers for face normals, bounds, and ray/triangle intersections.
- Procedural indexed meshes: cube, UV sphere, plane, and grid.
- A backend-independent event model and renderer contract.
- An initial Pygame backend for RGB pixel buffers, drawing lines, and input events.
- A small entity-component-system (ECS) registry, component data types, update-system contract, and synchronous event bus.
- A reusable RGB framebuffer and z-buffer with triangle and line rasterization.
- Initial frustum and back-face culling, homogeneous clipping, flat/Gouraud shading, and wireframe rendering.
- Transform and input systems that integrate the ECS with graphics and backend-independent events.
- A scene facade and fluent entity handles for common entity, camera, light, and transform operations.
- A portable OBJ loader that triangulates polygon faces into NumPy arrays.
- An engine loop with delta time, priority-ordered systems, event handling, and pause/resume/stop controls.
- Automatically discoverable unit tests for math, backend, core, graphics, API, assets, and engine modules.

## Requirements

- Python 3.10–3.14
- NumPy
- pygame-ce (for the Pygame-compatible backend)

The math package and backend interfaces can be imported without initializing Pygame. Pygame is loaded when `PygameBackend` is initialized.

## Getting started

Clone the repository and install Eko in editable mode with its Pygame-compatible backend:

```bash
git clone https://github.com/SuaveMolluscos/Eko.git
cd Eko
python -m pip install -e ".[pygame]"
```

To install dependencies from the requirements file instead, run `python -m pip install -r requirements.txt`. To install the test dependency as well as the package, use `python -m pip install -e ".[test]"`.

Run the interactive cube example from the repository root:

```bash
python examples/hello_cube.py
```

It opens an 800×600 Pygame window with a rotating cube. Close the window or press `Esc` to exit. The script locates the repository package relative to its own file, so direct execution works even when Python does not put the repository root on `sys.path`. For a bounded run, pass `--frames`, for example `python examples/hello_cube.py --frames 120`.

## Quick example

```python
from eko.mathematics import Quaternion, Transform, Vec3, cube

mesh = cube()
object_transform = Transform(
    position=Vec3(0, 0, 5),
    rotation=Quaternion.from_euler(yaw=0.5),
    scale=Vec3.one(),
)

world_matrix = object_transform.matrix
world_vertices = world_matrix.transform_points_batch(mesh.vertices)

print(f"Vertices: {len(world_vertices)}")
print(f"Triangles: {len(mesh.indices)}")
```

`Transform.matrix` is calculated on demand and cached until its position, rotation, or scale changes. The returned matrix is read-only; use the transform properties to update it.

## Backend example

The engine-facing code can depend on `IRenderer` and `EkoEvent` without using Pygame event constants:

```python
import numpy as np

from eko.backend import EventType, PygameBackend

renderer = PygameBackend(capture_mouse=False)
renderer.initialize(640, 480, "Eko")
try:
    framebuffer = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer.clear((15, 18, 24))
    renderer.blit_pixel_buffer(framebuffer)  # One full-buffer transfer per frame.
    renderer.present()

    for event in renderer.poll_events():
        if event.type is EventType.QUIT:
            print("Close requested")
        elif event.type is EventType.KEY_DOWN:
            print("Key pressed:", event.key)
finally:
    renderer.shutdown()
```

The framebuffer format is `(height, width, 3)` with `uint8` RGB values. Pygame converts it to the surface's `(width, height, 3)` layout before blitting. Mouse movement is reported as relative `(dx, dy)`; scroll is the vertical wheel delta.

## Core and ECS

`World` owns integer entity IDs and stores components by their concrete Python type. Queries start from the smallest component store and return components in the order requested:

```python
from eko import TransformComponent, Vec3, World

world = World()
entity = world.create_entity()
world.add_component(entity, TransformComponent(position=Vec3(0, 0, 5)))

for entity_id, transform in world.query(TransformComponent):
    print(entity_id, transform.position)
```

Adding a component of a type already present on an entity replaces it. `get_component()` returns `None` when the entity or component is absent; adding to or destroying an unknown entity raises `KeyError`. `get_all_components()` returns a shallow snapshot, so changing the returned dictionary does not modify the world.

`System` defines the `update(world, dt)` contract and stores `priority` and `enabled`; systems are not scheduled automatically yet. `EventBus` dispatches named events synchronously, in subscription order. Listener exceptions propagate to the emitter, and changes to subscriptions during an emission affect the next emission.

## Software graphics

The initial renderer is a CPU-based pipeline. It transforms meshes, culls world-space AABBs and back-facing triangles, clips against the view frustum, shades faces, and rasterizes into a reusable `PixelBuffer`. The Pygame backend receives the completed `(height, width, 3)` RGB array in one buffer transfer.

```python
from eko import (
    CameraComponent,
    MaterialComponent,
    MeshComponent,
    PixelBuffer,
    RenderSystem,
    TransformComponent,
    TransformSystem,
    Vec3,
    World,
    cube,
)
from eko.backend import PygameBackend

world = World()
camera = world.create_entity()
world.add_component(camera, CameraComponent(width=640, height=480))
world.add_component(camera, TransformComponent())

box = world.create_entity()
world.add_component(box, MeshComponent.from_mesh(cube()))
world.add_component(box, TransformComponent(position=Vec3(0, 0, 4)))
world.add_component(box, MaterialComponent(color=(220, 130, 70)))

transforms = TransformSystem()
renderer = RenderSystem(PixelBuffer(640, 480))
backend = PygameBackend()
backend.initialize(640, 480, "Eko")
try:
    transforms.update(world, 0.0)
    renderer.render(world, backend)  # Rasterize, blit once, and present.
finally:
    backend.shutdown()
```

`TransformSystem` should run before `RenderSystem`; it recalculates only dirty TRS matrices and updates attached mesh bounds. Components are plain data, so changing a transform after an update requires setting `transform.dirty = True`. `RenderSystem` currently supports one camera and a directional light supplied by its `light_dir` argument. `LightComponent` is data infrastructure and is not yet connected to the renderer. This is a first implementation: rasterization and per-face clipping still contain Python loops, while textures, transparency, and an automatic frame scheduler are not implemented.

`PixelBuffer` supports `FLAT`, `GOURAUD`, and `WIREFRAME` material modes. It owns its color and depth arrays and reuses them when cleared. `RenderSystem.render()` clears and renders the software buffer, then calls the backend's buffer blit and `present()`.

## Scene API and engine loop

`Scene` wraps a `World` with convenience methods. Entity handles provide chainable transform and material edits, while `scene.world` remains available when direct ECS access is needed:

```python
from eko import Engine, InputSystem, PixelBuffer, PygameBackend, RenderSystem
from eko import Scene, TransformSystem, cube
from eko.backend import Key

scene = Scene(width=640, height=480)
scene.create_camera(position=(0, 0, 0))
box = scene.create_entity(
    mesh=cube(),
    position=(0, 0, 4),
    color=(237, 33, 14),
)
box.rotate_y(-20).translate(1, 0, 0)

engine = Engine(
    scene.world,
    PygameBackend(),
    event_bus=scene.event_bus,
    title="Eko demo",
    target_fps=60,
)
engine.add_system(TransformSystem()).add_system(
    InputSystem({Key.W: "move_forward"})
).add_system(RenderSystem(PixelBuffer(640, 480)))
engine.run()
```

The engine initializes and shuts down the backend around `run()`. Each frame polls backend events, emits `backend_event` through the engine event bus, updates enabled systems in ascending `priority` order, blits the render system's pixel buffer once, and presents the frame. Delta time is measured in seconds with a monotonic high-resolution clock. `pause()`, `resume()`, and `stop()` can be called from another thread or from a system callback; pausing also interrupts the frame limiter, and paused time is excluded from the next delta. An OS quit event finishes the current frame and then exits. `run(max_frames=N)` is available for bounded applications and tests. With no render system, the engine presents a cleared black frame. The engine accepts explicit `width`/`height`; otherwise it uses the render buffer size, a scene camera size, or 1280×720.

`InputSystem(bindings)` accepts a mapping from `Key` values to action names. The engine attaches its event bus when registering the input system. Subscribe to actions through `scene.event_bus` when passing that bus to `Engine`; for example, `scene.event_bus.on("move_forward", callback)`. Key action callbacks receive `value`, `dt`, and `key`; key-down emits value `1.0` and key-up emits `0.0`.

`EntityHandle` uses radians for `rotate(x, y, z)` and degrees for the explicit convenience method `rotate_y(degrees)`. Its transform methods mark the component dirty so `TransformSystem` refreshes its matrix and bounds. Calling `destroy()` invalidates the handle; later operations that need a component raise an error rather than silently recreating the entity.

## OBJ assets

`load_obj(path)` accepts a string or `pathlib.Path` and returns `(vertices, faces)` with shapes `(V, 3)` and `(F, 3)`, using `float32` positions and `int32` triangle indices. Polygon faces are fan-triangulated. Position indices can be positive or relative negative indices; optional texture/normal index fields are accepted but not loaded. Materials, UV coordinates, normals, and smoothing groups are outside the loader's current scope.

When those arrays are passed to `Scene.create_entity()`, the resulting `MeshComponent` computes and stores its local AABB for the transform/culling pipeline.

```python
from pathlib import Path

from eko import Scene, load_obj

vertices, faces = load_obj(Path("assets") / "model.obj")
scene = Scene()
model = scene.create_entity(mesh=(vertices, faces), position=(0, 0, 5))
```

## Mathematics API

The main public API can be imported from `eko.mathematics`:

| Type or function | Purpose |
| --- | --- |
| `Vec3` / `vec3` | 3D vectors, dot and cross products, normalization, and arithmetic. |
| `Mat4` / `mat4` | 4x4 `float32` matrices, transforms, projections, cameras, and inversion. |
| `Quaternion` | Rotations, axis-angle and Euler construction, `slerp()`, and matrix conversion. |
| `Transform` | Cached TRS transform composed as translation × rotation × scale. |
| `Frustum` | Extracts six clip planes and tests AABBs against a view-projection matrix. |
| `Mesh` | Immutable indexed triangle mesh data with positions, normals, and indices. |
| `face_normal()` | Normalized normal for a triangle; returns zero for a degenerate triangle. |
| `aabb_from_vertices()` | Returns the minimum and maximum bounds of a vertex set. |
| `ray_triangle_intersect()` | Returns the ray parameter `t` for a hit, or `None`. |
| `cube()`, `sphere()`, `plane()`, `grid()` | Generates procedural triangle meshes without external model files. |

Lowercase `vec3` and `mat4` remain available inside `eko.mathematics` for the original module API. `Vec3` and `Mat4` are aliases, not separate implementations.

The backend API is imported separately from `eko.backend`:

```python
from eko.backend import EkoEvent, EventType, IRenderer, Key, PygameBackend
```

There are no top-level `mathematics` or `backend` compatibility packages. Use the `eko.*` import paths for all project code.

The core API can be imported from `eko.core` or from the top-level `eko` package:

```python
from eko.core import CameraComponent, EventBus, System, World
```

## Coordinate and transform conventions

These conventions are shared by the math modules and should be followed by future engine systems:

- Matrices are stored as NumPy `float32` arrays and multiply column vectors.
- Translation is in the last matrix column. `A @ B` applies `B` first, then `A`.
- World up is `+Y`; world forward is `+Z`. View space looks down `-Z`.
- Angles are in radians by default. APIs that accept `from_degrees=True` also accept degrees.
- Euler rotations use yaw about Y, pitch about X, and roll about Z, composed as `yaw * pitch * roll`.
- Quaternions are stored in `(w, x, y, z)` order.
- TRS matrices are composed as `T @ R @ S`.
- Perspective projection uses right-handed clip space and normalized depth in `[-1, 1]`.
- Points use homogeneous `w=1`; directions use `w=0`. `mat4 @ vec3` transforms a point and performs the perspective divide. Use `transform_homogeneous()` to retain clip-space `w`, or `transform_direction()` to ignore translation.
- `plane()` and `grid()` lie in the XZ plane and have `+Y` normals.
- `ray_triangle_intersect()` returns ray parameter `t`, not necessarily world-space distance. It is a distance only if the ray direction is normalized.
- `vec3` uses approximate equality and is unhashable.

## Running the tests

Install the test extra and run the full suite from the repository root:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -p "test_*.py" -v
```

## Repository layout

```text
eko/
  __init__.py          Top-level public API
  api/                 Scene facade and entity handles
  assets/              OBJ loader
  engine.py            Backend lifecycle and main loop
  mathematics/         Vectors, matrices, transforms, geometry, primitives
  backend/             Backend-independent events and renderer adapters
  core/                ECS registry, components, systems, event bus
  graphics/            Software framebuffer, culling, shaders, and systems
tests/                 Unit tests
```

## Roadmap

1. Stabilize the public API and add runnable examples with sample assets.
2. Improve input state, camera controls, and system scheduling ergonomics.
3. Expand the software renderer and asset pipeline through focused tests.
4. Evaluate native rendering backends only after the Python architecture has been validated.

The roadmap is exploratory; priorities may change as the project develops.

## Contributing

Ideas, bug reports, and focused pull requests are welcome. For changes to the math core, please include tests for the relevant behavior and preserve the documented coordinate conventions.

## License

Eko Engine is released under the [MIT License](./LICENSE).
