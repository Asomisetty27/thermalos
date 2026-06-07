"""Theta — GPU thermal-power forensics agent."""
from importlib.metadata import PackageNotFoundError, version as _v

try:
    __version__ = _v("runtheta")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
