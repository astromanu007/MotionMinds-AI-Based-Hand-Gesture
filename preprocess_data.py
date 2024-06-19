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
