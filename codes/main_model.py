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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import preprocessing 



train_data_gen = ImageDataGenerator(rotation_range=35,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest',
                                     )

valid_data_gen = ImageDataGenerator(rotation_range=35,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest',                                   
                                    )

test_data_gen = ImageDataGenerator()

dataset_dir = 'C:/Users/merti/Desktop/Marble_Class/train_marble/'
valid_dr = 'C:/Users/merti/Desktop/Marble_Class/validation_marble/'
test_dr = 'C:/Users/merti/Desktop/Marble_Class/test_marble/'

def convertTFLite(model):
      converter = tf.lite.TFLiteConverter.from_keras_model(model)
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      tflite_model = converter.convert() # Convert the model.
      with open('model.tflite', 'wb') as f: # Save the model.
         f.write(tflite_model)

 
Batch_size = 16
img_h = 224
img_w = 224
num_classes=25
classes = [ 'AageanRose','AfyonBal', 'AfyonBeyaz', 'AfyonBlack', 'AfyonGrey', 'AfyonSeker','Bejmermer',
                        'Blue','Capuchino', 'Diyabaz', 'DolceVita', 'EkvatorPijama', 'ElazigVisne','GoldGalaxy',
                        'GulKurusu','KaplanPostu', 'Karacabeysiyah','Konglomera', 'KristalEmprador',
                        'Leylakmermer', 'MediBlack', 'OliviaMarble','Oniks',
                        'RainGrey','Traverten',
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
                                             batch_size=16, 
                                             shuffle=False,
                                             seed=SEED,
                                             class_mode=None,
                                             )

class_names = np.array(['AageanRose','AfyonBal', 'AfyonBeyaz', 'AfyonBlack', 'AfyonGrey', 'AfyonSeker','Bejmermer',
                        'Blue','Capuchino', 'Diyabaz', 'DolceVita', 'EkvatorPijama', 'ElazigVisne','GoldGalaxy',
                        'GulKurusu','KaplanPostu', 'Karacabeysiyah','Konglomera', 'KristalEmprador',
                        'Leylakmermer', 'MediBlack', 'OliviaMarble','Oniks',
                        'RainGrey','Traverten'], dtype='<U10')


fig, c_ax = plt.subplots(1,1, figsize = (12, 8))

# function for scoring roc auc score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    target = ['AageanRose','AfyonBal', 'AfyonBeyaz', 'AfyonBlack', 'AfyonGrey', 'AfyonSeker','Bejmermer',
                        'Blue','Capuchino', 'Diyabaz', 'DolceVita', 'EkvatorPijama', 'ElazigVisne','GoldGalaxy',
                        'GulKurusu','KaplanPostu', 'Karacabeysiyah','Konglomera', 'KristalEmprador',
                        'Leylakmermer', 'MediBlack', 'OliviaMarble','Oniks',
                        'RainGrey','Traverten']  
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    return roc_auc_score(y_test, y_pred, average=average)
    
    
#


""" 
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(25,20))
  
  for n in range(0,7):
      ax = plt.subplot(1,7,n+1)
      plt.imshow(image_batch[n])
      plt.title(class_names[label_batch[n]==True][0].title())
      plt.axis('off')
      
image_batch, label_batch = next(train_gen)
show_batch(image_batch, label_batch)
"""

ResNet_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))

# The last 15 layers fine tune
for layer in ResNet_model.layers[:-15]:
    layer.trainable = False
tf.keras.layers.Rescaling(1./255)
x = ResNet_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(units=512,)(x)
x = Dropout(0.4)(x)
x = Dense(units=512,)(x)
x = Dropout(0.4)(x)
output  = Dense(units=25, activation='softmax')(x)
model = Model(ResNet_model.input, output)


model.summary()



print(len(model.layers))

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])


lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)


callbacks = [lrr]


tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    verbose=1,
    mode="min",
   
)

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
transfer_learning_history = model.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_gen,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=45,
                  callbacks=callbacks,)
                 
                  
acc = transfer_learning_history.history['accuracy']
val_acc = transfer_learning_history.history['val_accuracy']

loss = transfer_learning_history.history['loss']
val_loss = transfer_learning_history.history['val_loss']


epochs_range = range(45)

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


#model.evaluate(valid_gen, steps=STEP_SIZE_VALID,verbose=1)

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_gen, 3203 // Batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_gen.classes, y_pred))
print('Classification Report')
target_names = ['AageanRose','AfyonBal', 'AfyonBeyaz', 'AfyonBlack', 'AfyonGrey', 'AfyonSeker','Bejmermer',
                        'Blue','Capuchino', 'Diyabaz', 'DolceVita', 'EkvatorPijama', 'ElazigVisne','GoldGalaxy',
                        'GulKurusu','KaplanPostu', 'Karacabeysiyah','Konglomera', 'KristalEmprador',
                        'Leylakmermer', 'MediBlack', 'OliviaMarble','Oniks',
                        'RainGrey','Traverten']
print(classification_report(test_gen.classes, y_pred, target_names=target_names))
multiclass_roc_auc_score(test_gen.classes, y_pred)

plt.show()



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
  f = i[0:25]
  FN.append(f)
 
print(len(FN)) 
FN = FN[:len(predictions)]
print(len(FN))
results=pd.DataFrame({"Id":FN, "Category":predictions})

print(results)

convertTFLite(model)
