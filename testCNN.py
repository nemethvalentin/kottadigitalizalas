import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

test_img = cv2.imread('train_samples/validation/trebleClef/symbol116121.png')
test_gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
test_thresh = cv2.threshold(test_gray,100,255,cv2.THRESH_BINARY)[1]

model = keras.models.load_model('symbol_classification.h5')
test_img=cv2.resize(test_img, (120, 60), interpolation=cv2.INTER_CUBIC)
test_img=test_img.reshape(-1, 120, 60, 1)
prediction = np.argmax(model.predict(test_img)[0])
print(prediction)