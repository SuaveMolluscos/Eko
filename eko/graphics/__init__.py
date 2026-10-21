"""Software-rendering buffers, culling helpers, and graphics systems."""

from eko.graphics.culler import backface_cull, frustum_cull
from eko.graphics.pixel_buffer import PixelBuffer
from eko.graphics.shaders.flat import flat_shade
from eko.graphics.shaders.wireframe import wireframe_edges
from eko.graphics.systems.input_system import InputSystem
from eko.graphics.systems.render_system import RenderSystem
from eko.graphics.systems.transform_system import TransformSystem

__all__ = [
    "InputSystem",
    "PixelBuffer",
    "RenderSystem",
    "TransformSystem",
    "backface_cull",
    "flat_shade",
    "frustum_cull",
    "wireframe_edges",
]
