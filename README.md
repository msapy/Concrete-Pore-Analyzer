# 🏗️ Concrete Pore Analyzer

**Concrete Pore Analyzer** is a comprehensive Python application for advanced analysis of concrete pore structures using image processing.  
It provides an intuitive **GUI** for loading concrete images, segmenting pores, and extracting detailed quantitative metrics — including **shape, size, and spatial porosity distribution**.  

---

## ✨ Features

- **Multiple Detection Methods**  
  - Adaptive thresholding  
  - Edge detection  
  - Sample-based color segmentation  

- **Region of Interest (ROI)**  
  - Interactive ROI selection for localized pore analysis  

- **Scale Calibration**  
  - Draw scale lines for real-world unit conversion (mm / mm²)  

- **Advanced Analysis**  
  - Shape factor, pore size, and spatial porosity statistics  
  - Publication-ready visualizations  

- **Batch Export**  
  - Save all results, figures, and reports to a selected folder  

- **Professional Visualization**  
  - Large, readable fonts  
  - Modern **Seaborn** styling  

- **Interactive GUI**  
  - Tabbed interface for easy navigation and parameter adjustment  

---

## 📖 Usage

1. Load an image of a concrete specimen.  
2. Select an ROI and/or draw a scale line for calibration.  
3. Adjust detection parameters and choose a segmentation method.  
4. Analyze pores and view results in real-time.  
5. Export results (images, figures, and a detailed report).  

---

## 📊 Output

- Original image  
- Pore detection result  
- Binary pore mask  
- Shape factor histogram  
- Shape factor vs. area plot  
- Pore area distribution  
- Pore diameter distribution  
- Cumulative size distribution  
- Spatial porosity heatmap  
- 3D porosity visualization  
- Porosity distribution histogram  
- **Comprehensive analysis report**  

---

## ⚙️ Requirements

- Python 3.7+  
- [OpenCV](https://opencv.org/) (`cv2`)  
- [Pillow](https://python-pillow.org/) (`PIL`)  
- [Matplotlib](https://matplotlib.org/)  
- [Seaborn](https://seaborn.pydata.org/)  
- [Tkinter](https://docs.python.org/3/library/tkinter.html) (included with Python)  

Install dependencies with:  

```bash
pip install opencv-python pillow matplotlib seaborn
