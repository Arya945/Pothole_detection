# Pothole Detection

A YOLO-based computer vision project for detecting potholes in road images.

🏆 **Top 20 Project — AI For Youth 2019**

## Overview

This project uses **YOLO (You Only Look Once)** for pothole detection. The system processes an input image and identifies potholes using a trained YOLOv4 model.

## Project Structure

```text
pothole_detection/
├── detector.py
├── main.py
├── files/
│   ├── pothole.cfg
│   ├── obj.data
│   └── data.names
├── README.md
└── ...
```

## Model Weights

The YOLOv4 model weights are not included directly in the repository due to their large file size.

Download `yolov4.weights` from the **Releases** section and place it in the project directory.

```text
pothole_detection/
├── yolov4.weights
├── detector.py
├── main.py
└── files/
```

## Requirements

* Python 3.x
* OpenCV
* NumPy

Install the dependencies:

```bash
pip install opencv-python numpy
```

## Usage

Run:

```bash
python main.py
```

The program performs pothole detection using the YOLOv4 model and generates the detection result.

## Achievement

🏆 **Top 20 Project — AI For Youth 2019**

## Author

**Arya Pangging**
