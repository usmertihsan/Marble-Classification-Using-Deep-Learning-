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
from keras.applications import resnet


train_data_gen = ImageDataGenerator(rotation_range=30,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    cval=0,
                                    rescale=1./255)
valid_data_gen = ImageDataGenerator(rotation_range=35,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    cval=0,
                                    rescale=1./255)

test_data_gen = ImageDataGenerator(rescale=1./255)

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



def show_batch(image_batch, label_batch):
  plt.figure(figsize=(25,20))
  
  for n in range(0,6):
      ax = plt.subplot(1,6,n+1)
      plt.imshow(image_batch[n])
      plt.title(class_names[label_batch[n]==True][0].title())
      plt.axis('off')
      
image_batch, label_batch = next(train_gen)
show_batch(image_batch, label_batch)


resnet50_base = resnet.ResNet50(include_top=False, input_tensor=None, input_shape=(img_h, img_w,3))

tf.keras.layers.Rescaling(1./255)
output = resnet50_base.get_layer(index = -1).output  
output = Flatten()(output)
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(6, activation='softmax')(output)
resnet50_model = Model(resnet50_base.input, output)
for layer in resnet50_model.layers[:-7]:
    layer.trainable = False
resnet50_model.summary()
print(len(resnet50_model.layers))
resnet50_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics =['accuracy'])



STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
transfer_learning_history = resnet50_model.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_gen,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=30,
                    )               
                  
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

resnet50_model.evaluate(valid_gen, steps=STEP_SIZE_VALID,verbose=1)

STEP_SIZE_TEST=test_gen.n//test_gen.batch_size
test_gen.reset()
pred=resnet50_model.predict(test_gen,
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
FN = FN[:50]
print(len(FN))
results=pd.DataFrame({"Id":FN, "Category":predictions})

print(results)
