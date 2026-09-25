"""Native Rust implementation bundled with the UniqToken distribution."""

from .uniqtoken_core import *  # noqa: F403
from . import uniqtoken_core as _native

__doc__ = _native.__doc__
__version__ = "1.0.0"

if hasattr(_native, "__all__"):
    __all__ = _native.__all__
