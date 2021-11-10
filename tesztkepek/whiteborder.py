import numpy as np
import cv2

c = 1
while c <= 23:
    img = cv2.imread("template"+str(c)+".jpg")

    c=c+1