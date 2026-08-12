# 🔍 Object Detection Computer Vision

A computer vision application that detects objects in images using a pre-trained **MobileNet-SSD** deep learning model and **OpenCV**. The project provides an interactive **Streamlit web interface** where users can upload an image and view detected objects, confidence scores, and bounding boxes.

---

## 📌 Project Overview

This project demonstrates how a pre-trained object detection model can be integrated into a Python application to identify and locate objects inside images.

The system uses **MobileNet-SSD**, a lightweight object detection model designed for efficient inference, making it suitable for applications where computational resources are limited.

Users can:

- Upload an image
- Run object detection
- View the original image
- View the detected image
- See bounding boxes around detected objects
- View object labels
- View confidence scores
- View detection coordinates
- Analyze multiple detected objects

---

## 🎯 Project Objective

The main objective of this project is to understand and implement the complete computer vision inference pipeline:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Convert Image to Blob
     ↓
MobileNet-SSD Model
     ↓
Object Detection
     ↓
Confidence Filtering
     ↓
Bounding Box Generation
     ↓
Detection Visualization
     ↓
Streamlit Interface