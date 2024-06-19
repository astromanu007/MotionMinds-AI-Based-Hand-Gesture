# PRODIGY_ML_04

# Hand Gesture Recognition Model

## Overview

This project aims to develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data. The model enables intuitive human-computer interaction and gesture-based control systems.

## Features

- Recognizes and classifies various hand gestures.
- Supports both image and video data input for real-time recognition.
- Provides an interface for intuitive human-computer interaction.
- Can be integrated into gesture-based control systems.

## Prerequisites

- Python 3
- Required Python libraries: OpenCV, TensorFlow/Keras, numpy, matplotlib

## Getting Started

1. Clone this repository to your local machine:
   ```
   git clone https://github.com/your_username/hand-gesture-recognition.git
   ```

2. Install the required Python libraries:
   ```
   pip install opencv-python tensorflow numpy matplotlib
   ```

3. Download or capture the hand gesture dataset for training the model.

## Training the Model

1. Preprocess the dataset by resizing images, converting to grayscale, and normalizing pixel values.

2. Design and train a deep learning model using TensorFlow/Keras for hand gesture recognition. Experiment with different architectures such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) for optimal performance.

3. Evaluate the model's performance using validation data and adjust hyperparameters as needed.

## Testing the Model

1. Load the trained model weights.

2. Capture or provide input images/videos containing hand gestures.

3. Preprocess the input data similar to training data preprocessing.

4. Use the trained model to predict and classify hand gestures in real-time or batch processing.

## Conclusion

This project provides a comprehensive framework for developing and deploying a hand gesture recognition model. Experiment with different datasets, model architectures, and training techniques to achieve accurate and reliable gesture recognition for various applications.



To run the hand gesture recognition model, follow the instructions below. Make sure you have all the necessary libraries installed and your dataset prepared.

## How to Run

### 1. Clone the Repository

### 2. Install Dependencies
Install the required libraries using pip.
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

### 3. Prepare Your Dataset
Place your images in a structured directory format. For example:
```
dataset/
    gesture_0/
        img1.jpg
        img2.jpg
        ...
    gesture_1/
        img1.jpg
        img2.jpg
        ...
    ...
```

### 4. Preprocess the Data
Create a Python script named `preprocess_data.py` to preprocess your images.
```python
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_images(image_dir, image_size=(64, 64)):
    images = []
    labels = []
    label_names = os.listdir(image_dir)
    label_dict = {label: idx for idx, label in enumerate(label_names)}
    
    for label in label_names:
        image_files = os.listdir(os.path.join(image_dir, label))
        for image_file in image_files:
            image_path = os.path.join(image_dir, label, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(label_dict[label])
    
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels, label_dict

image_dir = 'dataset'
images, labels, label_dict = preprocess_images(image_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('label_dict.npy', label_dict)
```
Run the script to preprocess the data:
```bash
python preprocess_data.py
```

### 5. Train the Model
Create a Python script named `train_model.py` to train your CNN model.
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
label_dict = np.load('label_dict.npy', allow_pickle=True).item()

input_shape = (64, 64, 3)
num_classes = len(label_dict)
model = create_cnn_model(input_shape, num_classes)

history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

model.save('gesture_model.h5')
```
Run the script to train the model:
```bash
python train_model.py
```

### 6. Evaluate the Model
Create a Python script named `evaluate_model.py` to evaluate the model's performance.
```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('gesture_model.h5')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
```
Run the script to evaluate the model:
```bash
python evaluate_model.py
```

### 7. Predict Hand Gestures
Create a Python script named `predict_gesture.py` to use the trained model to predict hand gestures from new images.
```python
import tensorflow as tf
import cv2
import numpy as np

def predict_gesture(model, image_path, label_dict):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return np.argmax(prediction), label_dict

model = tf.keras.models.load_model('gesture_model.h5')
label_dict = np.load('label_dict.npy', allow_pickle=True).item()
inverse_label_dict = {v: k for k, v in label_dict.items()}

# Example prediction
image_path = 'path/to/new/image.jpg'
predicted_label, _ = predict_gesture(model, image_path, inverse_label_dict)
print(f'Predicted gesture: {inverse_label_dict[predicted_label]}')
```
Run the script to predict a gesture:
```bash
python predict_gesture.py
```

By following these steps, you can set up, train, evaluate, and use your hand gesture recognition model. Adjust the paths and parameters as needed for your specific dataset and requirements.
