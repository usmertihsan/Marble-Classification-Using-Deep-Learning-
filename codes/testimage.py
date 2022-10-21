import numpy as np
import cv2
import tensorflow as tf
img = cv2.imread("C:/Users/merti/Desktop/Marble_Class/afbeyaz1.jpg")
img = cv2.resize(img, (224,224))
img = np.array(img, dtype="float32")
img = np.reshape(img, (1,224,224,3))
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test the model on random input data.
input_shape = input_details[0]['shape']
print("*"*50, input_details)
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
output_probs = tf.math.softmax(output_data)
pred_label = tf.math.argmax(output_probs)
print(output_probs)

