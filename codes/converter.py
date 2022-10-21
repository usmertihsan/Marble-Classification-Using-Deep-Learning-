import tensorflow as tf

model = tf.keras.models.load_model('model.h5', compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.post_training_quantize = True
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)

