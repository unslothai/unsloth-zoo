# Unsloth Zoo MLX utilities.
#
# Keep this package initializer lightweight. The loader and trainer modules
# import MLX libraries and should stay lazy for non-MLX import paths.

from .runtime import is_mlx_available

__all__ = [
    "is_mlx_available",
]
