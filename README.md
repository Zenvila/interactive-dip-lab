# 🖼️ Digital Image Processing Learning Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)
![License](https://img.shields.io/badge/License-Educational-purple.svg)

**An interactive web-based simulator for visualizing and learning Digital Image Processing (DIP) algorithms**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API](#-api-endpoints)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Screenshots](#-screenshots)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

The **Digital Image Processing Learning Dashboard** is a comprehensive, interactive web application designed to help Computer Science students visualize and understand various DIP algorithms. Built with Flask and OpenCV, it provides step-by-step visualizations of complex image processing operations with real-time parameter adjustment.

### Key Highlights

- 🎨 **Modern, Attractive UI** with gradient designs and smooth animations
- 📊 **Interactive Visualizations** with real-time parameter adjustment
- 🧠 **Step-by-Step Simulations** for complex algorithms
- 🚀 **RESTful API** for all processing operations
- 📱 **Responsive Design** works on all devices
- 🔍 **Educational Focus** with detailed explanations for each step

---

## ✨ Features

### 📏 Interpolation (Scaling)
- Compare **Nearest Neighbor** vs **Bilinear Interpolation**
- Adjustable zoom factor (1x to 5x)
- Side-by-side visualization of original and processed images
- Real-time parameter adjustment

### 📊 Histogram Processing
- **Histogram Equalization**: Improve image contrast automatically
- **Histogram Matching**: Match one image's histogram to a reference image
- Visual histogram plots for comparison
- Dual image upload for matching

### 🔍 Canny Edge Detection
- **Step-by-step visualization**:
  - Step 1: Gaussian Blur (noise reduction)
  - Step 2: Gradient Computation (Sobel operators)
  - Step 3: Hysteresis Thresholding
- Interactive parameter controls for each step
- Expandable sections for detailed learning

### ⭐ Feature Detection
- **Harris Corner Detection**: Detect corners with adjustable parameters
  - Block size, k-value, and threshold controls
- **SIFT Keypoints**: Detect scale-invariant features
  - Maximum features control
  - Requires opencv-contrib-python

### 🧠 CNN (Convolutional Neural Network)
- **Step-by-step CNN visualization**:
  - **Convolution**: Apply different filters (Edge, Blur, Sharpen, Emboss)
  - **Activation**: ReLU, Sigmoid, Tanh functions
  - **Pooling**: Max and Average pooling
- Multiple feature maps visualization
- Interactive kernel and parameter controls

---

## 🛠️ Technologies Used

- **Backend**: Flask 3.0+
- **Image Processing**: OpenCV 4.8+
- **Numerical Computing**: NumPy 1.24+
- **Visualization**: Matplotlib 3.7+
- **Image Handling**: Pillow 10.0+
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5.3
- **Icons**: Font Awesome 6.4

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Zenvila/interactive-dip-lab.git
cd interactive-dip-lab
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- OpenCV (image processing)
- NumPy (numerical computations)
- Matplotlib (plotting)
- Pillow (image handling)
- Werkzeug (WSGI utilities)

### Step 3: (Optional) Install SIFT Support

For SIFT feature detection functionality:

```bash
pip install opencv-contrib-python
```

**Note:** If you already have `opencv-python` installed, uninstall it first:
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

---

## 🚀 Quick Start

### Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   - Navigate to `http://localhost:5000`
   - The dashboard will be ready to use!

3. **Upload an image:**
   - Click on any topic from the navigation menu
   - Click "Upload Image" or "Load Image"
   - Select an image file (PNG, JPG, JPEG, BMP, TIFF)

4. **Adjust parameters:**
   - Use sliders and dropdowns to adjust parameters
   - Results update in real-time
   - Compare original vs processed images side-by-side

### Running on Different Port

If port 5000 is already in use, modify `app.py`:

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Change port here
```

---

## 📖 Usage Guide

### Getting Started

1. **Navigate to a Topic**
   - Use the top navigation menu to select a topic
   - Each topic has its own dedicated page

2. **Upload an Image**
   - Supported formats: PNG, JPG, JPEG, BMP, TIFF
   - Maximum file size: 16MB
   - Click the upload button and select your image

3. **Adjust Parameters**
   - Use interactive controls (sliders, dropdowns)
   - Watch results update in real-time
   - Compare original and processed images

4. **Explore Step-by-Step**
   - For Canny Edge Detection and CNN, expand each step
   - Read explanations for each operation
   - Understand the algorithm flow

### Tips for Best Results

- **Image Size**: Smaller images (< 2MB) process faster
- **Image Format**: PNG and JPG work best
- **Parameter Tuning**: Start with default values, then adjust gradually
- **Multiple Images**: Try different images to see algorithm behavior
- **Step-by-Step**: For complex algorithms, go through each step sequentially

---

## 📁 Project Structure

```
interactive-dip-lab/
├── app.py                    # Main Flask application (routes & API)
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── DOCUMENTATION.md         # Comprehensive documentation
│
├── processors/              # Image processing logic
│   ├── __init__.py
│   ├── interpolation.py     # Scaling algorithms
│   ├── histogram.py        # Histogram operations
│   ├── canny.py            # Edge detection
│   ├── features.py         # Feature detection
│   └── cnn.py              # CNN operations
│
├── templates/               # HTML templates
│   ├── base.html           # Base template with navigation
│   ├── home.html           # Home page
│   ├── interpolation.html  # Interpolation page
│   ├── histogram.html      # Histogram page
│   ├── canny.html          # Canny edge detection
│   ├── features.html       # Feature detection
│   └── cnn.html            # CNN visualization
│
└── static/                  # Static files
    ├── css/
    │   └── style.css       # Custom styles
    └── js/
        └── utils.js        # Utility functions
```

---

## 🔌 API Endpoints

All processing operations are available via REST API endpoints.

### Base URL
```
http://localhost:5000
```

### Available Endpoints

#### Image Processing
- `POST /api/process/interpolation` - Image scaling
- `POST /api/process/histogram/equalize` - Histogram equalization
- `POST /api/process/histogram/match` - Histogram matching

#### Edge Detection
- `POST /api/process/canny/blur` - Gaussian blur step
- `POST /api/process/canny/gradients` - Gradient computation
- `POST /api/process/canny/threshold` - Hysteresis thresholding

#### Feature Detection
- `POST /api/process/features/harris` - Harris corner detection
- `POST /api/process/features/sift` - SIFT keypoint detection

#### CNN Operations
- `POST /api/process/cnn/convolution` - Convolution operation
- `POST /api/process/cnn/activation` - Activation function
- `POST /api/process/cnn/pooling` - Pooling operation
- `POST /api/process/cnn/feature_maps` - Multiple feature maps

### Example API Request

```bash
curl -X POST http://localhost:5000/api/process/interpolation \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image",
    "method": "bilinear",
    "zoom_factor": 2.0
  }'
```

For detailed API documentation, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## 📸 Screenshots

### Home Page
The dashboard home page provides an overview of all available topics with easy navigation.

### Interactive Processing
Each topic features interactive controls for real-time parameter adjustment and visualization.

### Step-by-Step Visualization
Complex algorithms like Canny Edge Detection and CNN are broken down into expandable steps with detailed explanations.

---

## 📚 Documentation

For comprehensive documentation, including:
- Detailed algorithm explanations
- Complete API reference
- Troubleshooting guide
- Architecture overview

See **[DOCUMENTATION.md](DOCUMENTATION.md)**

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

---

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'flask'"
**Solution:**
```bash
pip install -r requirements.txt
```

#### "SIFT not available"
**Solution:**
```bash
pip install opencv-contrib-python
```

#### Port 5000 already in use
**Solution:** Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

#### Images not processing
**Check:**
- Image format (PNG, JPG, JPEG, BMP, TIFF)
- File size (< 16MB)
- Browser console for errors (F12)

#### Slow processing
**Solutions:**
- Use smaller images
- Close other applications
- Check system resources


---

## 📄 License

This project is created for **educational purposes**.

© 2026 All Rights Reserved | Built by **Haris Shahzad**

---

## 👤 Author

**Haris Shahzad**

- **GitHub**: [@Zenvila](https://github.com/Zenvila)
- **LinkedIn**: [haris-shahzad-7b8746291](https://linkedin.com/in/haris-shahzad786)
- **Email**: [arainharis151@gmail.com](mailto:arainharis151@gmail.com)
- **Portfolio**: [Zenvila.github.io](https://Zenvila.github.io/)

---

## 🙏 Acknowledgments

- OpenCV community for excellent image processing library
- Flask team for the web framework
- Bootstrap for UI components
- Font Awesome for icons
- All contributors and users of this project

---

## ⭐ Show Your Support

If you find this project helpful, please give it a ⭐ on GitHub!

---

<div align="center">

**Made with dedication for Computer Science students learning Digital Image Processing**

[⬆ Back to Top](#-digital-image-processing-learning-dashboard)

</div>

