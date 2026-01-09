"""
Feature Detection Module
Implements Harris Corner Detection and SIFT keypoint detection.
"""

import cv2
import numpy as np


class FeatureDetector:
    """Detects features in images using various algorithms."""
    
    @staticmethod
    def harris_corners(image, block_size, k_value, threshold=0.01):
        """
        Detect corners using Harris Corner Detection.
        
        Args:
            image: Input image
            block_size: Size of neighborhood for corner detection
            k_value: Harris detector free parameter (typically 0.04-0.06)
            threshold: Threshold for corner detection (0.01 = 1% of max response)
            
        Returns:
            Image with corners marked, corner coordinates
        """
        if image is None or image.size == 0:
            return None, None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Convert to float32
        gray = np.float32(gray)
        
        # Apply Harris Corner Detection
        corners = cv2.cornerHarris(gray, block_size, 3, k_value)
        
        # Normalize and threshold
        corners = cv2.dilate(corners, None)
        
        # Create output image
        output = image.copy()
        
        # Mark corners
        corner_coords = []
        threshold_value = threshold * corners.max()
        corner_mask = corners > threshold_value
        
        # Draw circles on corners
        y_coords, x_coords = np.where(corner_mask)
        for x, y in zip(x_coords, y_coords):
            cv2.circle(output, (x, y), 5, (0, 0, 255), 2)
            corner_coords.append((x, y))
        
        return output, corner_coords
    
    @staticmethod
    def detect_sift(image, max_features=100):
        """
        Detect keypoints using SIFT (Scale-Invariant Feature Transform).
        
        Args:
            image: Input image
            max_features: Maximum number of features to detect
            
        Returns:
            Image with keypoints drawn, keypoints, descriptors
        """
        if image is None or image.size == 0:
            return None, None, None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        try:
            # Try to create SIFT detector
            # In newer OpenCV versions, SIFT might be in opencv-contrib-python
            # or accessed via cv2.SIFT_create() or cv2.xfeatures2d.SIFT_create()
            try:
                sift = cv2.SIFT_create(nfeatures=max_features)
            except AttributeError:
                # Fallback for older OpenCV versions
                try:
                    sift = cv2.xfeatures2d.SIFT_create(nfeatures=max_features)
                except AttributeError:
                    # If SIFT is not available, return None
                    return None, None, None
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            # Draw keypoints on image
            output = cv2.drawKeypoints(
                image, 
                keypoints, 
                None, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            return output, keypoints, descriptors
            
        except Exception as e:
            # Return None if SIFT is not available
            return None, None, None
