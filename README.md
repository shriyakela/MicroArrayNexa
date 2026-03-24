# MicroarrayNexa

**AI-Powered Microarray Spot Detection and Analysis System**

---

## 📋 Project Information

### Team Members
- **Shriya Kela** - 60009220219
- **Tanisha Chavan** - 60009220086
- **Vidhi Parmar** - 60009220054

---
## Demo Link : https://drive.google.com/file/d/1Xt-3JGEuyEnXHaS-UBVFtx448jTX_JBH/view?usp=drive_link

## 🎯 Problem Statement

### Challenge
Microarray technology is widely used in genomics research for analyzing gene expression patterns. However, manual detection and analysis of microarray spots is:
- **Time-consuming**: Requires manual inspection of thousands of spots
- **Error-prone**: Human interpretation is subjective and inconsistent
- **Labor-intensive**: Demands significant expertise and effort
- **Scalability issues**: Difficult to process large batches of images consistently

### Need
There is a critical need for an automated, accurate, and scalable solution that can:
1. Automatically detect and segment spots in microarray images
2. Calculate intensity measurements for each spot
3. Perform statistical analysis on the data
4. Generate comprehensive visualizations for interpretation
5. Export results for further downstream analysis

---

## 💡 Solution

### Overview
**MicroarrayNexa** is an intelligent system that combines deep learning with statistical analysis to automate microarray spot detection and comprehensive analysis.

### Key Features

#### 1. **AI-Powered Spot Detection**
- Uses a U-Net convolutional neural network trained on microarray images
- Employs semantic segmentation to precisely identify and segment individual spots
- Achieves high accuracy in spot boundary detection
- Robust to variations in image quality, intensity, and background

#### 2. **Intensity Analysis**
- Calculates mean intensity for each detected spot
- Normalizes intensities using Z-score standardization
- Groups spots into microarray regions for comparative analysis
- Exports detailed spot-level and region-level statistics

#### 3. **Statistical Visualizations**
- **Density Plot**: Visualizes distribution of normalized intensities
- **Box Plot**: Shows quartile distribution across regions
- **Heatmap**: Displays spatial intensity patterns with color gradients
- **Intensity Bar Chart**: Region-wise mean intensity with error bars

#### 4. **User-Friendly Web Interface**
- Modern Flask-based web application
- Dark-themed responsive design
- Real-time image upload and processing
- One-click visualization access in new tabs
- CSV export functionality for further analysis

#### 5. **Automated CSV Export**
- Structured output with spot number and mean intensity
- Compatible with downstream analysis tools
- Easy integration with R, Python, or other statistical software

---

## 🛠️ Technology Stack

### Backend
| Technology | Purpose | Version |
|-----------|---------|---------|
| **Python** | Core programming language | 3.8+ |
| **Flask** | Web framework | Latest |
| **PyTorch** | Deep learning framework | Latest |
| **U-Net** | Neural network architecture | Custom implementation |

### Data Processing & Analysis
| Technology | Purpose |
|-----------|---------|
| **NumPy** | Numerical computations |
| **SciPy** | Scientific computing & image processing |
| **Pandas** | Data manipulation and analysis |
| **OpenCV** | Image processing utilities |
| **Pillow (PIL)** | Image I/O and manipulation |

### Visualization & Frontend
| Technology | Purpose |
|-----------|---------|
| **Matplotlib** | Statistical plot generation |
| **Seaborn** | Statistical data visualization |
| **Bootstrap 5** | Responsive CSS framework |
| **HTML5 & CSS3** | Web interface markup & styling |
| **JavaScript** | Client-side interactivity |



## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser

### Installation

1. **Clone or download the project**
   ```bash
   cd spotrixui
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open in web browser**
   ```
   http://localhost:5000
   ```

3. **Upload and analyze**
   - Click "Select Image" to upload a microarray image
   - Click "Analyze Image" to process
   - View results with statistics and visualizations

---

## 📊 How It Works

### Processing Pipeline

```
User Upload
    ↓
Image Preprocessing
(Convert to grayscale, resize to 256×256, normalize)
    ↓
U-Net Model Inference
(Semantic segmentation for spot detection)
    ↓
Spot Identification
(Connected component labeling, boundary detection)
    ↓
Intensity Calculation
(Mean intensity per spot)
    ↓
Visualization Generation
(Density plot, boxplot, heatmap, intensity chart)
    ↓
Results Display & Export
(Statistics table, CSV download, visualization tabs)
```




## 📝 Workflow Example

### Step 1: Upload Image
```
User selects microarray image file
System validates format and prepares for processing
```

### Step 2: Analyze
```
Image → U-Net Model → Spot Detection → Intensity Calculation
```

### Step 3: View Results
```
Statistics displayed in main tab
Visualizations button opens new tab with 4 plots
CSV button downloads detailed results
```

### Step 4: Export (Optional)
```
Click "Download CSV"
Save spots.csv for external analysis
```

---

## 🔧 Dependencies

All dependencies are listed in `requirements.txt`:

```
flask              # Web framework
torch              # Deep learning
torchvision        # Computer vision utilities
opencv-python      # Image processing
numpy              # Numerical computing
scipy              # Scientific computing
Pillow             # Image I/O
matplotlib         # Plotting library
seaborn            # Statistical visualization
pandas             # Data manipulation
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📌 Important Notes

### Model Location
- Ensure `unet_trained_state.pth` is in the project root directory
- Model is loaded on application startup
- Requires ~50MB of disk space

### Input Requirements
- **Format**: PNG or JPEG images
- **Size**: Will be resized to 256×256 pixels
- **Channels**: Converted to grayscale internally
- **Quality**: Best results with clear, high-contrast microarray images

### Limitations
- Processes one image at a time
- Batch processing not yet implemented
- Requires adequate RAM for visualization generation
- Process time: ~2-5 seconds per image

---

## 🎓 Educational Use

This project demonstrates:
- **Deep Learning**: U-Net architecture for semantic segmentation
- **Web Development**: Flask-based full-stack application
- **Image Processing**: Preprocessing, filtering, and analysis
- **Data Visualization**: Statistical plot generation
- **Software Engineering**: Modular design and error handling

---

## 🤝 Contributing

This is an academic project. For improvements or bug reports:
1. Document the issue clearly
2. Provide sample images if applicable
3. Include error logs and system information

---

## 📄 License

This project was developed as coursework for a major project. Use of this code is restricted to educational purposes.

---

## 🔗 References & Resources

### Deep Learning
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- PyTorch Documentation: https://pytorch.org/docs/

### Image Processing
- OpenCV Documentation: https://docs.opencv.org/
- SciPy ndimage Module: https://docs.scipy.org/doc/scipy/reference/ndimage.html

### Microarray Analysis
- Gene expression microarray protocols
- Spot intensity normalization techniques
- Statistical analysis of microarray data

---

## 👥 Contact & Support

For questions regarding this project, please contact the development team:
- Shriya Kela (60009220219)
- Tanisha Chavan (60009220086)
- Vidhi Parmar (60009220054)

---

## ✨ Version History

### Version 1.0 (Current)
- ✅ U-Net spot detection model
- ✅ Automated intensity analysis
- ✅ Four-visualization system (density, box, heatmap, intensity)
- ✅ Web-based user interface
- ✅ CSV export functionality
- ✅ Statistical analysis and reporting

---

## 🎯 Future Enhancements

Potential improvements for future versions:
- [ ] Batch image processing
- [ ] GPU acceleration support
- [ ] Advanced filtering options
- [ ] Model retraining interface
- [ ] Extended statistical tests
- [ ] Multi-format export (Excel, PDF)
- [ ] Cloud deployment capabilities
- [ ] Mobile app version
- [ ] Real-time image preview
- [ ] Comparison analysis tools

---

**Last Updated**: March 2026

**Status**: ✅ Production Ready

---

*MicroarrayNexa: Bringing Intelligence to Microarray Analysis*
