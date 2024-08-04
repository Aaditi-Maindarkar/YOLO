# YOLOv8 Object Detection

This repository contains code for training, predicting, and real-time prediction using the YOLOv8 model. The project utilizes the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library for object detection tasks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Detection Train](#detection-train)
  - [Predict](#predict)
  - [Real-Time Predict](#real-time-predict)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/yolov8-object-detection.git
   cd yolov8-object-detection
   
## Install the required packages
```bash
ultralytics
opencv-python
matplotlib
numpy
```

## Usage

1. **Detection Train**:
To train a YOLOv8 model, use the following code:

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```
2. **Predict**:
To make predictions on an image, use the following code:
```bash
from ultralytics import YOLO

# Load the model
model = YOLO("C:/YOLO/yolov8n.pt")  # load an official model
model = YOLO("C:/YOLO/runs/detect/train/weights/best.pt")  # load a custom model

# Predict with the model
results = model("C:/YOLO/runs/detect/images/elephant.jpg")  # predict on an image

# Process and visualize results
for result in results:
    result.show()  # display the image with predictions
    result.save(save_dir='C:/YOLO/runs/detect/predictions')  # save the image with predictions

print("Prediction complete. Check the predictions directory for results.")
```
3. **Real-Time Predict**:
To perform real-time prediction using a webcam, use the following code:
```bash
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("C:/YOLO/yolov8n.pt")  # load an official model
# model = YOLO("C:/YOLO/runs/detect/train/weights/best.pt")  # load a custom model

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Predict with the model
        results = model(frame)
        
        # Process and visualize results
        for result in results:
            # Plot the results on the frame
            annotated_frame = result.plot()
        
        # Convert BGR to RGB
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the resulting frame
        ax.clear()
        ax.imshow(annotated_frame_rgb)
        ax.axis('off')
        plt.draw()
        plt.pause(0.001)
        
        # Check for 'q' key press to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the capture and close the plot
    cap.release()
    plt.close()
```
## Dataset
For training the model, you will need a dataset in the format specified by YOLO. The dataset configuration file should be specified in the data argument of the model.train() method. An example configuration file, such as coco8.yaml, can be used.

## Results
After training and making predictions, results will be saved in the specified directories. For real-time predictions, the processed frames will be displayed using Matplotlib.

