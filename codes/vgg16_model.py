import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras import Model 
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D,BatchNormalization
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint

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


""" 
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(25,20))
  
  for n in range(0,6):
      ax = plt.subplot(1,6,n+1)
      plt.imshow(image_batch[n])
      plt.title(class_names[label_batch[n]==True][0].title())
      plt.axis('off')
      
image_batch, label_batch = next(train_gen)
show_batch(image_batch, label_batch)

"""

callbacks = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
best_model_file = 'C:/Users/merti/Desktop/Marble_Class/vgg16_drop_batch_best_weights_256.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)
reduce_lr = ReduceLROnPlateau(patience=5, monitor='val_acc', factor=0.1, min_lr=0.0000001, mode='auto', verbose=1)


wp = 'C:/Users/merti/Desktop/Marble_Class/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg16_base = VGG16(include_top=False, weights=wp,
                   input_tensor=None, input_shape=(224, 224, 3))

output = vgg16_base.get_layer(index = -1).output  
output = Flatten()(output)
# let's add a fully-connected layer
output = Dense(1024,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(1024,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(6, activation='softmax')(output)
vgg16_model = Model(vgg16_base.input, output)
for layer in vgg16_model.layers[:-7]:
    layer.trainable = False
vgg16_model.summary()
vgg16_model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics =['accuracy'])


history = vgg16_model.fit_generator(train_gen,
                              epochs=1,
                              verbose=1,
                              validation_data=valid_gen,
                              callbacks = [callbacks, best_model]
                              )

acc=history.history['accuracy']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))


fig = plt.figure(figsize=(20,10))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.show()

""" 



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
                   epochs=3,
                  callbacks=callbacks,)
                 
                  
acc = transfer_learning_history.history['accuracy']
val_acc = transfer_learning_history.history['val_accuracy']

loss = transfer_learning_history.history['loss']
val_loss = transfer_learning_history.history['val_loss']


epochs_range = range(3)

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
"""

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(vgg16_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)