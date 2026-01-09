"""
CNN (Convolutional Neural Network) Module
Step-by-step visualization of CNN operations.
"""

import cv2
import numpy as np


class CNNProcessor:
    """Processes CNN operations step by step."""
    
    @staticmethod
    def create_conv_kernel(kernel_type='edge', size=3):
        """
        Create different types of convolution kernels.
        
        Args:
            kernel_type: Type of kernel ('edge', 'blur', 'sharpen', 'emboss')
            size: Size of the kernel (must be odd)
            
        Returns:
            Convolution kernel
        """
        if size % 2 == 0:
            size += 1
        
        if kernel_type == 'edge':
            # Edge detection kernel (Sobel-like)
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=np.float32)
        elif kernel_type == 'blur':
            # Blur kernel (average)
            kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        elif kernel_type == 'sharpen':
            # Sharpen kernel
            kernel = np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]], dtype=np.float32)
        elif kernel_type == 'emboss':
            # Emboss kernel
            kernel = np.array([[-2, -1,  0],
                              [-1,  1,  1],
                              [ 0,  1,  2]], dtype=np.float32)
        else:
            # Default: identity
            kernel = np.zeros((size, size), dtype=np.float32)
            kernel[size//2, size//2] = 1.0
        
        return kernel
    
    @staticmethod
    def apply_convolution(image, kernel):
        """
        Apply convolution operation to image.
        
        Args:
            image: Input image
            kernel: Convolution kernel
            
        Returns:
            Convolved image
        """
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply convolution
        result = cv2.filter2D(gray, -1, kernel)
        
        return result
    
    @staticmethod
    def apply_pooling(image, pool_type='max', pool_size=2):
        """
        Apply pooling operation (Max or Average).
        
        Args:
            image: Input image
            pool_type: Type of pooling ('max' or 'average')
            pool_size: Size of pooling window
            
        Returns:
            Pooled image
        """
        if image is None or image.size == 0:
            return None
        
        h, w = image.shape[:2]
        new_h = h // pool_size
        new_w = w // pool_size
        
        if pool_type == 'max':
            pooled = np.zeros((new_h, new_w), dtype=image.dtype)
            for i in range(new_h):
                for j in range(new_w):
                    y1 = i * pool_size
                    y2 = min((i + 1) * pool_size, h)
                    x1 = j * pool_size
                    x2 = min((j + 1) * pool_size, w)
                    pooled[i, j] = np.max(image[y1:y2, x1:x2])
        else:  # average
            pooled = np.zeros((new_h, new_w), dtype=image.dtype)
            for i in range(new_h):
                for j in range(new_w):
                    y1 = i * pool_size
                    y2 = min((i + 1) * pool_size, h)
                    x1 = j * pool_size
                    x2 = min((j + 1) * pool_size, w)
                    pooled[i, j] = np.mean(image[y1:y2, x1:x2])
        
        return pooled
    
    @staticmethod
    def apply_activation(image, activation_type='relu'):
        """
        Apply activation function.
        
        Args:
            image: Input image
            activation_type: Type of activation ('relu', 'sigmoid', 'tanh')
            
        Returns:
            Activated image
        """
        if image is None or image.size == 0:
            return None
        
        # Normalize to 0-1 range
        img_float = image.astype(np.float32) / 255.0
        
        if activation_type == 'relu':
            # ReLU: max(0, x)
            activated = np.maximum(0, img_float)
        elif activation_type == 'sigmoid':
            # Sigmoid: 1 / (1 + exp(-x))
            activated = 1.0 / (1.0 + np.exp(-img_float))
        elif activation_type == 'tanh':
            # Tanh: tanh(x)
            activated = np.tanh(img_float)
        else:
            activated = img_float
        
        # Convert back to 0-255 range
        activated = (activated * 255).astype(np.uint8)
        
        return activated
    
    @staticmethod
    def visualize_feature_maps(image, num_filters=6):
        """
        Generate multiple feature maps using different kernels.
        
        Args:
            image: Input image
            num_filters: Number of different filters to apply
            
        Returns:
            List of feature maps
        """
        if image is None or image.size == 0:
            return []
        
        kernels = [
            ('edge', 'Edge Detection'),
            ('blur', 'Blur'),
            ('sharpen', 'Sharpen'),
            ('emboss', 'Emboss'),
        ]
        
        feature_maps = []
        for i, (kernel_type, name) in enumerate(kernels[:num_filters]):
            kernel = CNNProcessor.create_conv_kernel(kernel_type)
            feature_map = CNNProcessor.apply_convolution(image, kernel)
            if feature_map is not None:
                feature_maps.append({
                    'name': name,
                    'image': feature_map,
                    'kernel_type': kernel_type
                })
        
        return feature_maps
