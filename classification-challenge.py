import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy

batch_size = 32

# Create an instance of ImageDataGenerator for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    #Randomly increase or decrease the size of the image by up to 10%
    zoom_range = 0.1, 
    #Randomly rotate the image between -25,25 degrees
    rotation_range = 25, 
    #Shift the image along its width by up to +/- 5%
    width_shift_range = 0.05, 
    #Shift the image along its height by up to +/- 5%
    height_shift_range = 0.05,
    )

# Create a generator for the training set
train_generator = train_datagen.flow_from_directory(
    '/Users/thear/Documents/Code/x_ray/classification-challenge/classification-challenge-starter/Covid19-dataset/train',
    target_size = (256, 256),
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = 'grayscale')

# train_generator.next()

# Create a generator for the validation set
val_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = val_datagen.flow_from_directory(
    '/Users/thear/Documents/Code/x_ray/classification-challenge/classification-challenge-starter/Covid19-dataset/train',
    target_size = (256, 256),
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = 'grayscale')

#Build model
model = Sequential()

model.add(tf.keras.Input(shape = (256,256,1)))

model.add(tf.keras.layers.Conv2D(5, 5, strides = 3, activation = 'relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(3, 3, strides = 1, activation = 'relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(layers.Dropout(0.1))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(3, activation = 'softmax'))

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = tf.keras.losses.CategoricalCrossentropy(), 
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

# early stopping implementation
es = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=20)

print(model.summary())

history = model.fit(
    train_generator, 
    steps_per_epoch = train_generator.samples/batch_size,
    epochs = 8,
    validation_data = val_generator,
    validation_steps = val_generator.samples/batch_size,
    callbacks = [es]
    )

# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc_6'])
ax2.plot(history.history['val_auc_6'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()


test_steps_per_epoch = numpy.math.ceil(val_generator.samples / val_generator.batch_size)

predictions = model.predict(val_generator, 
                            steps = test_steps_per_epoch)

test_steps_per_epoch = numpy.math.ceil(val_generator.samples / val_generator.batch_size)

predicted_classes = numpy.argmax(predictions,
                                 axis = 1)

true_classes = val_generator.classes

class_labels = list(val_generator.class_indices.keys())

report = classification_report(true_classes, 
                               predicted_classes, 
                               target_names = class_labels)

print(report)   
 
cm=confusion_matrix(true_classes,predicted_classes)

print(cm)







