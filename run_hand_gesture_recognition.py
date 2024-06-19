import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

def train_model(X_train, y_train, input_shape, num_classes):
    model = create_cnn_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)
    model.save('gesture_model.h5')
    return model

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}')

def predict_gesture(model, image_path, label_dict):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return np.argmax(prediction), label_dict

def main():
    # Set the image directory
    image_dir = 'dataset'
    
    # Preprocess the data
    print("Preprocessing the data...")
    images, labels, label_dict = preprocess_images(image_dir)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training the model...")
    input_shape = (64, 64, 3)
    num_classes = len(label_dict)
    model = train_model(X_train, y_train, input_shape, num_classes)
    
    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    
    # Predict a gesture
    print("Predicting a gesture...")
    image_path = 'path/to/new/image.jpg'  # Replace with the path to your image
    predicted_label, _ = predict_gesture(model, image_path, {v: k for k, v in label_dict.items()})
    print(f'Predicted gesture: {predicted_label}')

if __name__ == "__main__":
    main()
