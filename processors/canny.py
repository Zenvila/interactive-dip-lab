"""
Canny Edge Detection Module
Implements step-by-step Canny edge detection with visualization of each stage.
"""

import cv2
import numpy as np


class CannyProcessor:
    """Processes Canny edge detection step by step."""
    
    @staticmethod
    def gaussian_blur(image, kernel_size, sigma):
        """
        Apply Gaussian blur to reduce noise.
        
        Args:
            image: Input image
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation of Gaussian kernel
            
        Returns:
            Blurred image
        """
        if image is None or image.size == 0:
            return None
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def compute_gradients(image):
        """
        Compute gradient magnitude and direction using Sobel operators.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Gradient magnitude, gradient direction (in degrees)
        """
        if image is None or image.size == 0:
            return None, None
        
        # Compute gradients in x and y directions
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Normalize magnitude to 0-255
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        return magnitude, direction
    
    @staticmethod
    def hysteresis_thresholding(image, low_threshold, high_threshold):
        """
        Apply hysteresis thresholding to create binary edge map.
        
        Args:
            image: Gradient magnitude image
            low_threshold: Lower threshold for weak edges
            high_threshold: Upper threshold for strong edges
            
        Returns:
            Binary edge map
        """
        if image is None or image.size == 0:
            return None
        
        # Normalize image to 0-255 if needed
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply thresholds
        strong_edges = (image >= high_threshold).astype(np.uint8) * 255
        weak_edges = ((image >= low_threshold) & (image < high_threshold)).astype(np.uint8) * 128
        
        # Combine strong and weak edges
        edges = strong_edges + weak_edges
        
        # Use OpenCV's Canny for final result (includes edge tracking)
        # This is a simplified version - full implementation would include edge tracking
        final_edges = cv2.Canny(image, low_threshold, high_threshold)
        
        return final_edges, edges
