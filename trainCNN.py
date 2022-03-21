import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.layers import Activation, Dense, Flatten, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os

train_path= 'train_samples/train'

train_batches = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    batch_size=13,
    image_size=(32,32),
    color_mode='grayscale',
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
)

valid_batches = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    batch_size=13,
    image_size=(32,32),
    color_mode='grayscale',
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
)

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomTranslation(0.1, 0.1)
    ])

#Szekvenciális modell létrehozása
model = Sequential()
model.add(data_augmentation)
model.add(layers.Input(shape=(32,32,1)))
model.add(Conv2D(16, 3, activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPool2D())
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(13, activation="softmax"))


#print(len(train_batches)) 
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.summary()
model.fit(train_batches, validation_data=valid_batches, epochs=20, verbose=2)

model.save("symbol_classification.h5")