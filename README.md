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
   ```bash
pip install ultralytics opencv-python matplotlib numpy

```
## Model Training
If you want to train the model on your own dataset, ensure your dataset is correctly formatted and labeled. You can start training by modifying the dataset configuration file and running the training script.

from ultralytics import YOLO

# Load the YOLOv8 model

```bash
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # Transfer learning from pretrained weights
```

# Train the model

```bash
results = model.train(data="path/to/your/dataset.yaml", epochs=100, imgsz=640)
```
data: Path to your dataset configuration file.
epochs: Number of training epochs.
imgsz: Image size for training.

# Making Predictions on Images
You can use the trained model to make predictions on new images. Make sure the image paths are correctly specified.

```bash
from ultralytics import YOLO
```

# Load the model
```bash
model = YOLO("path/to/your/model.pt")  # Load your trained model
```

# Predict on a new image
```bash
results = model("path/to/your/image.jpg")
```

# Process and visualize results
```
for result in results:
    result.show()  # Display the image with predictions
    result.save(save_dir='path/to/save/predictions')  # Save the annotated image
```
model: Path to the trained model file.
image.jpg: Path to the image you want to make predictions on.
save_dir: Directory where you want to save the annotated images.

## Real-Time Detection Using a Webcam
You can run real-time detection using a webcam by executing the following script:

```bash
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("path/to/your/model.pt")

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
            annotated_frame = result.plot()
        
        # Convert BGR to RGB for visualization
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
model.pt: Path to your trained model file.
cv2.VideoCapture(0): Initializes the webcam (0 for the default camera).
'q' key: Press to stop the real-time detection.
```

## Checking Results
Predictions: After running predictions, check the specified save directory for annotated images.
Training Logs: Review training logs and results to fine-tune model parameters if necessary.
Real-Time Detection: Observe real-time object detection and adjust model settings for optimal performance.

## Customizing the Project
Feel free to customize the project according to your needs. You can adjust the model architecture, dataset, or detection parameters as required.
