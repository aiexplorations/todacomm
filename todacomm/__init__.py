"""
ToDACoMM - Topological Data Analysis Comparison of Multiple Models

A pipeline for comparing topological properties of activation spaces across
different pre-trained transformer models using persistent homology.
"""

__version__ = "0.1.0"
__author__ = "Rajesh Sampathkumar"
__email__ = "rexplorations@gmail.com"

from . import models
from . import tda
from . import data
from . import extract
from . import analysis
from . import utils
from . import visualization

__all__ = [
    "models",
    "tda",
    "data",
    "extract",
    "analysis",
    "utils",
    "visualization",
]
