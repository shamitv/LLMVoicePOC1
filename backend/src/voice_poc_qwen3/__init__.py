"""voice_poc_qwen3 package

Installable package for the backend of the Voice POC.
"""

from .config import Config, config

__version__ = "0.1.0"

# Exported symbols (keep minimal for now)
__all__ = ["__version__", "Config", "config"]
