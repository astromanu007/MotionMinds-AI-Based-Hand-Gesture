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
