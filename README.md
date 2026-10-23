# Face & Body Scanner

Real-time and static-image human **Face Mesh**, **Body Pose**, and **Hand Landmark** scanner powered by **MediaPipe Holistic** and **OpenCV**.

## Features

- **Face Mesh Tracking**: 478 3D landmarks (including iris refinement)
- **Full-Body Pose Tracking**: 33 3D skeletal landmarks
- **Hand Tracking**: 21 landmarks per hand (left & right)
- **Dual Mode**: Real-time webcam stream with live HUD or static image processing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Real-Time Webcam Mode

```bash
python scanner.py
```

Press `q` or `ESC` to exit the webcam window.

### 2. Static Image Mode

```bash
python scanner.py --image path/to/photo.jpg --output output_scanned.jpg
```
