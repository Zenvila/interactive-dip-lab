"""
Image Transformations Module
Handles various intensity transformation operations.
"""

import cv2
import numpy as np


class TransformationProcessor:
    """Processes various image intensity transformations."""
    
    @staticmethod
    def image_negative(image):
        """
        Apply image negative transformation.
        Formula: s = L - 1 - r, where L is the number of intensity levels
        
        Args:
            image: Input image
            
        Returns:
            Negated image
        """
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply negative: s = 255 - r
        negative = 255 - gray
        
        return negative
    
    @staticmethod
    def log_transformation(image, c=1.0):
        """
        Apply logarithmic transformation.
        Formula: s = c * log(1 + r)
        Enhances dark regions of the image.
        
        Args:
            image: Input image
            c: Scaling constant (default: 1.0)
            
        Returns:
            Log-transformed image
        """
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Convert to float and normalize
        img_float = gray.astype(np.float32)
        
        # Apply log transformation: s = c * log(1 + r)
        log_transformed = c * np.log1p(img_float)
        
        # Normalize to 0-255 range
        log_transformed = (255 * log_transformed / np.max(log_transformed)).astype(np.uint8)
        
        return log_transformed
    
    @staticmethod
    def power_law_transformation(image, gamma=1.0, c=1.0):
        """
        Apply power-law (gamma) transformation.
        Formula: s = c * r^gamma
        Gamma < 1: Brightens dark regions
        Gamma > 1: Darkens bright regions
        
        Args:
            image: Input image
            gamma: Gamma value (default: 1.0)
            c: Scaling constant (default: 1.0)
            
        Returns:
            Power-law transformed image
        """
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range
        img_normalized = gray.astype(np.float32) / 255.0
        
        # Apply power-law: s = c * r^gamma
        power_law = c * np.power(img_normalized, gamma)
        
        # Convert back to 0-255 range
        power_law = (255 * power_law).astype(np.uint8)
        
        return power_law
    
    @staticmethod
    def bit_plane_slicing(image, bit_plane=7):
        """
        Extract a specific bit plane from the image.
        Each bit plane contains different information about the image.
        
        Args:
            image: Input image
            bit_plane: Which bit plane to extract (0-7, where 7 is MSB)
            
        Returns:
            Bit plane image
        """
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ensure bit_plane is in valid range
        bit_plane = max(0, min(7, int(bit_plane)))
        
        # Extract bit plane: (image >> bit_plane) & 1
        bit_plane_img = (gray >> bit_plane) & 1
        
        # Scale to 0-255 for better visualization
        bit_plane_img = bit_plane_img * 255
        
        return bit_plane_img.astype(np.uint8)
    
    @staticmethod
    def contrast_stretching(image, r1=0, s1=0, r2=255, s2=255):
        """
        Apply contrast stretching (piecewise linear transformation).
        Enhances contrast by stretching intensity range.
        
        Args:
            image: Input image
            r1, s1: First control point (input, output)
            r2, s2: Second control point (input, output)
            
        Returns:
            Contrast-stretched image
        """
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create lookup table
        lut = np.zeros(256, dtype=np.uint8)
        
        # Piecewise linear transformation
        for r in range(256):
            if r < r1:
                lut[r] = int(s1 * r / r1) if r1 > 0 else 0
            elif r < r2:
                lut[r] = int(s1 + (s2 - s1) * (r - r1) / (r2 - r1)) if r2 > r1 else s1
            else:
                lut[r] = int(s2 + (255 - s2) * (r - r2) / (255 - r2)) if r2 < 255 else s2
        
        # Apply transformation
        stretched = cv2.LUT(gray, lut)
        
        return stretched
    
    @staticmethod
    def histogram_stretching(image):
        """
        Apply histogram stretching (automatic contrast enhancement).
        Stretches the histogram to use full intensity range.
        
        Args:
            image: Input image
            
        Returns:
            Histogram-stretched image
        """
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Find min and max pixel values
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        # Avoid division by zero
        if max_val == min_val:
            return gray
        
        # Stretch to full range: s = (r - min) * 255 / (max - min)
        stretched = ((gray - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
        
        return stretched
    
    @staticmethod
    def thresholding(image, threshold=127, threshold_type='binary'):
        """
        Apply thresholding transformation.
        
        Args:
            image: Input image
            threshold: Threshold value (0-255)
            threshold_type: Type of thresholding ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv')
            
        Returns:
            Thresholded image
        """
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Map threshold type
        type_map = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV
        }
        
        thresh_type = type_map.get(threshold_type, cv2.THRESH_BINARY)
        _, thresholded = cv2.threshold(gray, threshold, 255, thresh_type)
        
        return thresholded
