"""ML data collection package."""

# Import the correct class names
from .stall_detector import StallDetector, StallDetectionConfig
from .feature_extractor import LayerFeatureExtractor, FeatureExtractionConfig

# Update __all__ to reflect the correct names
__all__ = [
    'StallDetector',
    'StallDetectionConfig',
    'LayerFeatureExtractor',
    'FeatureExtractionConfig'
]