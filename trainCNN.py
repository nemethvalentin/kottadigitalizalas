import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.layers import  Dense, Flatten, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix

train_path= 'train_samples/train_new'

train_batches = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    batch_size=10,
    image_size=(120,60),
    color_mode='grayscale',
    shuffle=True,
    seed=456,
    validation_split=0.2,
    subset="training",
)

valid_batches = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    batch_size=10,
    image_size=(120,60),
    color_mode='grayscale',
    shuffle=True,
    seed=456,
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
model.add(layers.Input(shape=(120,60,1)))
model.add(Conv2D(8, 3, activation="relu"))
model.add(MaxPool2D(2))
model.add(Conv2D(16, 3, activation="relu"))
model.add(MaxPool2D(2))
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPool2D(2))
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPool2D(2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


#print(len(train_batches)) 
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.summary()
model.fit(train_batches, validation_data=valid_batches, epochs=20, verbose=2)

model.save("symbol_classification.h5")