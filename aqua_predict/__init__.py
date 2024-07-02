import sys
import os


sys.path.append(os.path.dirname(__file__))
__all__ = [
    "config", "data", "fun", "main"
]

from .main import *