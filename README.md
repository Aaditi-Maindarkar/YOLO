# YOLO Object Detection Project

This project demonstrates the use of the YOLOv8 model for object detection tasks, including training a custom model, making predictions on images, and real-time detection using a webcam.

## Features

- **Model Training**: Train a YOLOv8 model on custom datasets.
- **Image Prediction**: Perform object detection on static images.
- **Real-Time Prediction**: Detect objects in real-time using a webcam feed.
- **Visualization**: Display and save annotated images with detected objects.

## Requirements

- Python 3.7+
- `ultralytics` (YOLO implementation)
- `opencv-python` (Computer vision tasks)
- `matplotlib` (Plotting)
- `numpy` (Numerical computations)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yolo-object-detection.git
   cd yolo-object-detection
   
Install the required packages:


pip install ultralytics opencv-python matplotlib numpy

Download the pre-trained YOLOv8 weights or train a custom model.

Usage
1. Model Training
Train the YOLOv8 model on your dataset. Customize the dataset path and other parameters as needed.

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

2. Image Prediction
Make predictions on images using a trained YOLO model.
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

3. Real-Time Prediction
Perform real-time object detection using a webcam.


import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("C:/YOLO/yolov8n.pt")  # load an official model

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
