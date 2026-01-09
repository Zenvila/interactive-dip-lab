"""
Histogram Processing Module
Handles histogram equalization and histogram matching operations.
"""

import cv2
import numpy as np


class HistogramProcessor:
    """Processes histogram operations on images."""
    
    @staticmethod
    def equalize_histogram(image):
        """
        Apply histogram equalization to improve image contrast.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            Equalized image, original histogram, equalized histogram
        """
        if image is None or image.size == 0:
            return None, None, None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate original histogram
        hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Calculate equalized histogram
        hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
        
        return equalized, hist_original, hist_equalized
    
    @staticmethod
    def match_histogram(source_image, reference_image):
        """
        Match the histogram of source image to reference image.
        
        Args:
            source_image: Source image to be transformed
            reference_image: Reference image whose histogram will be matched
            
        Returns:
            Matched image, source histogram, reference histogram, matched histogram
        """
        if source_image is None or source_image.size == 0:
            return None, None, None, None
        if reference_image is None or reference_image.size == 0:
            return None, None, None, None
        
        # Convert to grayscale if needed
        if len(source_image.shape) == 3:
            source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source_image.copy()
            
        if len(reference_image.shape) == 3:
            ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference_image.copy()
        
        # Calculate histograms
        hist_source = cv2.calcHist([source_gray], [0], None, [256], [0, 256])
        hist_ref = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
        
        # Calculate cumulative distribution functions
        cdf_source = hist_source.cumsum()
        cdf_ref = hist_ref.cumsum()
        
        # Normalize CDFs
        cdf_source = (cdf_source - cdf_source.min()) * 255 / (cdf_source.max() - cdf_source.min())
        cdf_ref = (cdf_ref - cdf_ref.min()) * 255 / (cdf_ref.max() - cdf_ref.min())
        
        # Create mapping function
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find the closest value in reference CDF
            idx = np.argmin(np.abs(cdf_ref - cdf_source[i]))
            mapping[i] = idx
        
        # Apply mapping
        matched = cv2.LUT(source_gray, mapping)
        
        # Calculate matched histogram
        hist_matched = cv2.calcHist([matched], [0], None, [256], [0, 256])
        
        return matched, hist_source, hist_ref, hist_matched
