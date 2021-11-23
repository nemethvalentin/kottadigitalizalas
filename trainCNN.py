from keras_preprocessing.image import image_data_generator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.layers import Activation, Dense, Flatten, MaxPool2D, Conv2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os

train_path= 'train_samples/train'
valid_path= 'train_samples/validation'
test_path= 'train_samples/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(32,32), classes=['altoClef', 'beams', 'flat', 'naturals', 'notes', 'notesFlags', 'notesOpen', 'relation', 'rests1', 'rests2', 'sharps', 'time', 'trebleClef', 'unknown'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(32,32), classes=['altoClef', 'beams', 'flat', 'naturals', 'notes', 'notesFlags', 'notesOpen', 'relation', 'rests1', 'rests2', 'sharps', 'time', 'trebleClef', 'unknown'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(32,32), classes=['altoClef', 'beams', 'flat', 'naturals', 'notes', 'notesFlags', 'notesOpen', 'relation', 'rests1', 'rests2', 'sharps', 'time', 'trebleClef', 'unknown'], batch_size=10)


imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)
print(labels)

#Szekvenciális modell létrehozása
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(14, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)

print(test_batches.classes)

predictions = model.predict(x=test_batches, verbose=0)
np.round(predictions)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

def plot_confusion_matrix(cm, classes, normalize=False, title= 'Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")

    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment= "center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(test_batches.class_indices)

cm_plot_labels= ['altoClef', 'beams', 'flat', 'naturals', 'notes', 'notesFlags', 'notesOpen', 'relation', 'rests1', 'rests2', 'sharps', 'time', 'trebleClef', 'unknown']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

if os.path.isfile('symbol_classification.h5') is False:
    model.save("symbol_classification.h5")