"""
Interpolation Module
Handles image scaling using Nearest Neighbor and Bilinear interpolation methods.
"""

import cv2
import numpy as np


class InterpolationProcessor:
    """Processes image scaling with different interpolation methods."""
    
    @staticmethod
    def nearest_neighbor(image, scale_factor):
        """
        Upscale image using Nearest Neighbor interpolation.
        
        Args:
            image: Input image (numpy array)
            scale_factor: Scaling factor (e.g., 2.0 for 2x zoom)
            
        Returns:
            Scaled image
        """
        if image is None or image.size == 0:
            return None
            
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    @staticmethod
    def bilinear(image, scale_factor):
        """
        Upscale image using Bilinear interpolation.
        
        Args:
            image: Input image (numpy array)
            scale_factor: Scaling factor (e.g., 2.0 for 2x zoom)
            
        Returns:
            Scaled image
        """
        if image is None or image.size == 0:
            return None
            
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
