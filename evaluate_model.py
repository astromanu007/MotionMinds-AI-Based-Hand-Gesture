import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('gesture_model.h5')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
