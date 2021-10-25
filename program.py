import cv2
import numpy as np

img = cv2.imread('kepek/test/test1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh= cv2.threshold(gray,100,255,cv2.THRESH_BINARY)[1]
thresh = 255-thresh
thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
minLineLength = 1000
maxLineGap = 10
lines = cv2.HoughLinesP(thresh,1,np.pi/2,1000,minLineLength,maxLineGap)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1), (x2,y2), (0,255,0), 2)

cv2.imwrite('linetest.jpg',img)
cv2.waitKey(0)