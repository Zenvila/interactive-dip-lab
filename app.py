"""
Digital Image Processing Learning Dashboard - Flask Application
Main Flask app with routes and API endpoints.
"""

import os
import base64
import io
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from processors import (
    InterpolationProcessor,
    HistogramProcessor,
    CannyProcessor,
    FeatureDetector,
    CNNProcessor,
    TransformationProcessor
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize processors
interpolation_processor = InterpolationProcessor()
histogram_processor = HistogramProcessor()
canny_processor = CannyProcessor()
feature_detector = FeatureDetector()
cnn_processor = CNNProcessor()
transformation_processor = TransformationProcessor()


def image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    if image is None:
        return None
    # Convert BGR to RGB
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', image_rgb)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


def load_image_from_base64(base64_string):
    """Load image from base64 string."""
    try:
        img_data = base64.b64decode(base64_string.split(',')[1] if ',' in base64_string else base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def create_histogram_plot(hist, title="Histogram", color='blue'):
    """Create histogram plot and return as base64."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist, color=color)
    ax.set_title(title)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


# ==================== Routes ====================

@app.route('/')
def home():
    """Home page."""
    return render_template('home.html')


@app.route('/interpolation')
def interpolation():
    """Interpolation page."""
    return render_template('interpolation.html')


@app.route('/histogram')
def histogram():
    """Histogram processing page."""
    return render_template('histogram.html')


@app.route('/canny')
def canny():
    """Canny edge detection page."""
    return render_template('canny.html')


@app.route('/features')
def features():
    """Feature detection page."""
    return render_template('features.html')


@app.route('/cnn')
def cnn():
    """CNN visualization page."""
    return render_template('cnn.html')


@app.route('/transformations')
def transformations():
    """Image transformations page."""
    return render_template('transformations.html')


# ==================== API Endpoints ====================

@app.route('/api/process/interpolation', methods=['POST'])
def api_interpolation():
    """API endpoint for interpolation."""
    try:
        data = request.json
        image_base64 = data.get('image')
        method = data.get('method', 'nearest')
        zoom_factor = float(data.get('zoom_factor', 2.0))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        if method == 'nearest':
            result = interpolation_processor.nearest_neighbor(image, zoom_factor)
        else:
            result = interpolation_processor.bilinear(image, zoom_factor)
        
        result_base64 = image_to_base64(result)
        
        return jsonify({
            'success': True,
            'result': result_base64,
            'original_size': f"{image.shape[1]}x{image.shape[0]}",
            'result_size': f"{result.shape[1]}x{result.shape[0]}" if result is not None else "N/A"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/histogram/equalize', methods=['POST'])
def api_histogram_equalize():
    """API endpoint for histogram equalization."""
    try:
        data = request.json
        image_base64 = data.get('image')
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        equalized, hist_orig, hist_eq = histogram_processor.equalize_histogram(image)
        
        return jsonify({
            'success': True,
            'equalized': image_to_base64(equalized),
            'hist_original': create_histogram_plot(hist_orig, "Original Histogram", 'blue'),
            'hist_equalized': create_histogram_plot(hist_eq, "Equalized Histogram", 'green')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/histogram/match', methods=['POST'])
def api_histogram_match():
    """API endpoint for histogram matching."""
    try:
        data = request.json
        source_base64 = data.get('source_image')
        reference_base64 = data.get('reference_image')
        
        source = load_image_from_base64(source_base64)
        reference = load_image_from_base64(reference_base64)
        
        if source is None or reference is None:
            return jsonify({'error': 'Invalid image(s)'}), 400
        
        matched, hist_source, hist_ref, hist_matched = histogram_processor.match_histogram(
            source, reference
        )
        
        return jsonify({
            'success': True,
            'matched': image_to_base64(matched),
            'hist_source': create_histogram_plot(hist_source, "Source", 'blue'),
            'hist_reference': create_histogram_plot(hist_ref, "Reference", 'orange'),
            'hist_matched': create_histogram_plot(hist_matched, "Matched", 'green')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/canny/blur', methods=['POST'])
def api_canny_blur():
    """API endpoint for Gaussian blur step."""
    try:
        data = request.json
        image_base64 = data.get('image')
        kernel_size = int(data.get('kernel_size', 5))
        sigma = float(data.get('sigma', 1.0))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        blurred = canny_processor.gaussian_blur(image, kernel_size, sigma)
        
        return jsonify({
            'success': True,
            'blurred': image_to_base64(blurred)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/canny/gradients', methods=['POST'])
def api_canny_gradients():
    """API endpoint for gradient computation step."""
    try:
        data = request.json
        image_base64 = data.get('image')
        kernel_size = int(data.get('kernel_size', 5))
        sigma = float(data.get('sigma', 1.0))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # First blur
        blurred = canny_processor.gaussian_blur(image, kernel_size, sigma)
        magnitude, direction = canny_processor.compute_gradients(blurred)
        
        # Normalize direction for display
        direction_normalized = ((direction + 180) / 360 * 255).astype(np.uint8)
        
        return jsonify({
            'success': True,
            'magnitude': image_to_base64(magnitude),
            'direction': image_to_base64(direction_normalized)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/canny/threshold', methods=['POST'])
def api_canny_threshold():
    """API endpoint for hysteresis thresholding step."""
    try:
        data = request.json
        image_base64 = data.get('image')
        kernel_size = int(data.get('kernel_size', 5))
        sigma = float(data.get('sigma', 1.0))
        low_threshold = int(data.get('low_threshold', 50))
        high_threshold = int(data.get('high_threshold', 150))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Process through all steps
        blurred = canny_processor.gaussian_blur(image, kernel_size, sigma)
        final_edges, thresholded = canny_processor.hysteresis_thresholding(
            blurred, low_threshold, high_threshold
        )
        
        return jsonify({
            'success': True,
            'thresholded': image_to_base64(thresholded),
            'final_edges': image_to_base64(final_edges)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/features/harris', methods=['POST'])
def api_harris():
    """API endpoint for Harris corner detection."""
    try:
        data = request.json
        image_base64 = data.get('image')
        block_size = int(data.get('block_size', 2))
        k_value = float(data.get('k_value', 0.04))
        threshold = float(data.get('threshold', 0.01))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result, corners = feature_detector.harris_corners(image, block_size, k_value, threshold)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result),
            'corner_count': len(corners)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/features/sift', methods=['POST'])
def api_sift():
    """API endpoint for SIFT keypoint detection."""
    try:
        data = request.json
        image_base64 = data.get('image')
        max_features = int(data.get('max_features', 100))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result, keypoints, descriptors = feature_detector.detect_sift(image, max_features)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': 'SIFT is not available. Install opencv-contrib-python: pip install opencv-contrib-python'
            }), 400
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result),
            'keypoint_count': len(keypoints) if keypoints else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/cnn/convolution', methods=['POST'])
def api_cnn_convolution():
    """API endpoint for CNN convolution step."""
    try:
        data = request.json
        image_base64 = data.get('image')
        kernel_type = data.get('kernel_type', 'edge')
        kernel_size = int(data.get('kernel_size', 3))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        kernel = cnn_processor.create_conv_kernel(kernel_type, kernel_size)
        result = cnn_processor.apply_convolution(image, kernel)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result),
            'kernel': kernel.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/cnn/pooling', methods=['POST'])
def api_cnn_pooling():
    """API endpoint for CNN pooling step."""
    try:
        data = request.json
        image_base64 = data.get('image')
        pool_type = data.get('pool_type', 'max')
        pool_size = int(data.get('pool_size', 2))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = cnn_processor.apply_pooling(image, pool_type, pool_size)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result),
            'original_size': f"{image.shape[1]}x{image.shape[0]}",
            'result_size': f"{result.shape[1]}x{result.shape[0]}" if result is not None else "N/A"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/cnn/activation', methods=['POST'])
def api_cnn_activation():
    """API endpoint for CNN activation step."""
    try:
        data = request.json
        image_base64 = data.get('image')
        activation_type = data.get('activation_type', 'relu')
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = cnn_processor.apply_activation(image, activation_type)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/cnn/feature_maps', methods=['POST'])
def api_cnn_feature_maps():
    """API endpoint for CNN feature maps visualization."""
    try:
        data = request.json
        image_base64 = data.get('image')
        num_filters = int(data.get('num_filters', 4))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        feature_maps = cnn_processor.visualize_feature_maps(image, num_filters)
        
        result = []
        for fm in feature_maps:
            result.append({
                'name': fm['name'],
                'image': image_to_base64(fm['image']),
                'kernel_type': fm['kernel_type']
            })
        
        return jsonify({
            'success': True,
            'feature_maps': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/transformations/negative', methods=['POST'])
def api_negative():
    """API endpoint for image negative transformation."""
    try:
        data = request.json
        image_base64 = data.get('image')
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = transformation_processor.image_negative(image)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/transformations/log', methods=['POST'])
def api_log_transform():
    """API endpoint for logarithmic transformation."""
    try:
        data = request.json
        image_base64 = data.get('image')
        c = float(data.get('c', 1.0))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = transformation_processor.log_transformation(image, c)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/transformations/power_law', methods=['POST'])
def api_power_law():
    """API endpoint for power-law (gamma) transformation."""
    try:
        data = request.json
        image_base64 = data.get('image')
        gamma = float(data.get('gamma', 1.0))
        c = float(data.get('c', 1.0))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = transformation_processor.power_law_transformation(image, gamma, c)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/transformations/bit_plane', methods=['POST'])
def api_bit_plane():
    """API endpoint for bit plane slicing."""
    try:
        data = request.json
        image_base64 = data.get('image')
        bit_plane = int(data.get('bit_plane', 7))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = transformation_processor.bit_plane_slicing(image, bit_plane)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/transformations/contrast_stretch', methods=['POST'])
def api_contrast_stretch():
    """API endpoint for contrast stretching."""
    try:
        data = request.json
        image_base64 = data.get('image')
        r1 = int(data.get('r1', 0))
        s1 = int(data.get('s1', 0))
        r2 = int(data.get('r2', 255))
        s2 = int(data.get('s2', 255))
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = transformation_processor.contrast_stretching(image, r1, s1, r2, s2)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/transformations/histogram_stretch', methods=['POST'])
def api_histogram_stretch():
    """API endpoint for histogram stretching."""
    try:
        data = request.json
        image_base64 = data.get('image')
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = transformation_processor.histogram_stretching(image)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/transformations/threshold', methods=['POST'])
def api_threshold():
    """API endpoint for thresholding."""
    try:
        data = request.json
        image_base64 = data.get('image')
        threshold = int(data.get('threshold', 127))
        threshold_type = data.get('threshold_type', 'binary')
        
        image = load_image_from_base64(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        result = transformation_processor.thresholding(image, threshold, threshold_type)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
