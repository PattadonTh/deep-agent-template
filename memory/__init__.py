# This package exposes a single entry point: get_memory_store().
# Everywhere else in the codebase (main.py, etc.) should import from here
# rather than importing backend classes directly, so swapping backends
# only requires changing the MEMORY_BACKEND env var, not the calling code.
from .backends import get_memory_store

__all__ = ["get_memory_store"]
