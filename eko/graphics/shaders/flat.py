import numpy as np


def flat_shade(normals, light_dir, base_color, ambient=0.15):
    """Return uint8 per-face colors; light_dir points from the light into the scene."""
    normals = np.asarray(normals, dtype=np.float32)
    light_dir = np.asarray(light_dir, dtype=np.float32)
    base_color = np.asarray(base_color, dtype=np.float32)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (F, 3)")
    if light_dir.shape != (3,):
        raise ValueError("light_dir must have shape (3,)")
    if base_color.shape != (3,):
        raise ValueError("base_color must have shape (3,)")
    if (not np.all(np.isfinite(normals)) or not np.all(np.isfinite(light_dir))
            or not np.all(np.isfinite(base_color))):
        raise ValueError("shader inputs must be finite")
    if np.any(base_color < 0) or np.any(base_color > 255):
        raise ValueError("base_color channels must be in [0, 255]")
    if not np.isfinite(ambient) or not 0 <= ambient <= 1:
        raise ValueError("ambient must be in [0, 1]")

    light_length = np.linalg.norm(light_dir)
    if light_length <= 1e-12:
        raise ValueError("light_dir must not be zero")
    normalized_light = light_dir / light_length
    normal_lengths = np.linalg.norm(normals, axis=1)
    normalized_normals = np.divide(
        normals,
        normal_lengths[:, np.newaxis],
        out=np.zeros_like(normals),
        where=normal_lengths[:, np.newaxis] > 1e-12,
    )
    diffuse = np.clip(normalized_normals @ -normalized_light, 0.0, 1.0)
    intensity = ambient + (1.0 - ambient) * diffuse
    return np.clip(base_color[np.newaxis, :] * intensity[:, np.newaxis], 0, 255).astype(np.uint8)
