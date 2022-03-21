#Könyvtárak beimportálása
import cv2
import numpy as np
from tensorflow import keras

#Kép beolvasása és transzformálása
img = cv2.imread('kepek/tesztkepek/test1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh= cv2.threshold(gray,100,255,cv2.THRESH_BINARY)[1]
thresh = 255-thresh
thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

#Kottavonalak megkeresése
minLineLength = img.shape[1]*0.75
maxLineGap = 25
lines = cv2.HoughLinesP(thresh,1,np.pi/2,1000,minLineLength,maxLineGap)
goodLines = []

for line in lines:
   x1,y1,x2,y2 = line[0]
   if abs(x2-x1) > abs(y2-y1):
       goodLines.append(line[0])
            
goodLines.sort(key=lambda line: line[1])

permaLines = []
tempLines = []
i=0
counter = 0

while i<len(goodLines):
   tempLine=goodLines[i]
   beforeY = tempLine[1]
   j=i+1
   while j<len(goodLines) and goodLines[j][1]-beforeY <= 5:
       if abs(goodLines[j][2]-goodLines[j][0]) > abs(tempLine[2]-tempLine[0]) :
           tempLine = goodLines[j]
       j += 1
   
   tempLines.append(tempLine)
   i=j+1
   if len(tempLines) == 5:
       permaLines.append(tempLines)
       tempLines = []
       
#Kottavonalak kitörlése
horizontal = np.copy(thresh)
rows = horizontal.shape[0]
horizontalSize = 5
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, horizontalSize))
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

horizontal = cv2.bitwise_not(horizontal)

#Alakzatok kontúrjainak megkeresése és megjelölése
horizontal_reverse = 255-horizontal
contours, hierarchy = cv2.findContours(horizontal_reverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

horizontalRGB = cv2.cvtColor(horizontal,cv2.COLOR_GRAY2BGR)
cv2.drawContours(horizontalRGB, contours, -1, (0,0,255), thickness=2)

#Alakzatok eltárolása
symbols = []

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)

    cv2.rectangle(horizontalRGB, (x,y), (x+w, y+h), (255,0,255), thickness=2)

    symbols.append(horizontal[y:y+h, x:x+w])

symbols = sorted(symbols, key=lambda symbol: symbol.shape[1], reverse=True)

#Neurális modell alkalmazása
classified_symbols=[]
model = keras.models.load_model('symbol_classification.h5')
for x in symbols:
    x=cv2.resize(x, (32, 32), interpolation=cv2.INTER_CUBIC)
    x=x.reshape(-1, 32, 32, 1)
    prediction = model.predict(x)
    classified_symbols.append(np.argmax(prediction[0]))

cv2.imwrite("lines.jpg", lines)
cv2.imwrite("contours.jpg", horizontalRGB)
cv2.imwrite("linesremoved.jpg", horizontal)

cv2.waitKey(0)