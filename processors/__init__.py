"""
Digital Image Processing Processors
This package contains all image processing modules.
"""

from .interpolation import InterpolationProcessor
from .histogram import HistogramProcessor
from .canny import CannyProcessor
from .features import FeatureDetector
from .cnn import CNNProcessor
from .transformations import TransformationProcessor

__all__ = [
    'InterpolationProcessor',
    'HistogramProcessor',
    'CannyProcessor',
    'FeatureDetector',
    'CNNProcessor',
    'TransformationProcessor'
]
