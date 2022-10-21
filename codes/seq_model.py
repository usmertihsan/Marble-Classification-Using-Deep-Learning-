import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras import Model 
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
import datetime, os
import numpy
import keras
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, LeakyReLU,ELU
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


train_data_gen = ImageDataGenerator(rotation_range=30,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest',
                                    cval=0,
                                    )
valid_data_gen = ImageDataGenerator(rotation_range=35,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest',
                                    cval=0,
                                    )

test_data_gen = ImageDataGenerator()

dataset_dir = 'C:/Users/merti/Desktop/Marble_Class/train_marble/'
valid_dr = 'C:/Users/merti/Desktop/Marble_Class/validation_marble/'
test_dr = 'C:/Users/merti/Desktop/Marble_Class/test_marble/'

Batch_size = 8
img_h = 224
img_w = 224
num_classes=6
classes = [ 'AfyonBeyaz', # 0
            'AfyonGrey1', # 1
            'Bejmermer', # 2
            'Kaplanpostu', # 3
            'Karacabeysiyah', # 4
            'KristalEmprador',# 5
           ]


# Training
SEED = 1234
tf.random.set_seed(SEED) 


train_gen = train_data_gen.flow_from_directory(dataset_dir,
                                               target_size=(224, 224),
                                               batch_size=Batch_size,
                                               classes=classes,
                                               class_mode='categorical',
                                               #save_to_dir='C:/Users/merti/Desktop/Marble_Class/augmentation/',
                                               #save_prefix='',
                                               #save_format='jpg',
                                               shuffle=True,
                                               seed=SEED)  # targets are directly converted into one-hot vectors

# Validation
valid_gen = valid_data_gen.flow_from_directory(valid_dr,
                                           target_size=(224, 224),
                                           batch_size=Batch_size, 
                                           classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED)

test_gen = test_data_gen.flow_from_directory(test_dr,
                                             target_size=(224, 224),
                                             batch_size=10, 
                                             shuffle=False,
                                             seed=SEED,
                                             class_mode=None,
                                             )

class_names = np.array(['AfyonBeyaz','AfyonGrey1', 'Bejmermer', 'Kaplanpostu', 'Karacabeysiyah', 'KristalEmprador'], dtype='<U10')

model = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1./255),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation="relu"),
     tf.keras.layers.Dense(6, activation='softmax')
    ]
)

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)
callbacks = [lrr]

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
transfer_learning_history = model.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_gen,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=30,
                  callbacks=callbacks,)
                           
acc = transfer_learning_history.history['accuracy']
val_acc = transfer_learning_history.history['val_accuracy']

loss = transfer_learning_history.history['loss']
val_loss = transfer_learning_history.history['val_loss']


epochs_range = range(30)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.evaluate(valid_gen, steps=STEP_SIZE_VALID,verbose=1)

STEP_SIZE_TEST=test_gen.n//test_gen.batch_size
test_gen.reset()
pred=model.predict(test_gen,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
print(pred)
print(len(pred))
print(predicted_class_indices)
print(len(predicted_class_indices))
labels = train_gen.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [k for k in predicted_class_indices]
print(len(predictions))
filenames=test_gen.filenames
FN=[]
for i in filenames:
  f = i[0:6]
  FN.append(f)
 
print(len(FN)) 
FN = FN[:len(predictions)]
print(len(FN))
results=pd.DataFrame({"Id":FN, "Category":predictions})

print(results)


# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
  
  
""" 
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
"""