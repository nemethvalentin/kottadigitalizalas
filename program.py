#Könyvtárak beimportálása
import cv2
import numpy as np
from tensorflow import keras

from createOutput import create_midi

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

    symbols.append([horizontal[y:y+h, x:x+w], [x, y, w, h]])

symbols = sorted(symbols, key=lambda symbol: symbol[0].shape[1], reverse=True)

#Neurális modell alkalmazása
classified_symbols=[]
note_symbols_pos=[]
model = keras.models.load_model('symbol_classification.h5')
for symbolPic, symbolPos in symbols:
    symbolPic=cv2.resize(symbolPic, (120, 60), interpolation=cv2.INTER_CUBIC)
    symbolPic=symbolPic.reshape(-1, 120, 60, 1)
    prediction= model.predict(symbolPic)
    classified_symbols.append(np.argmax(prediction[0]))
    predicted = np.argmax(prediction[0])
    # if predicted == 2 or predicted == 3 or predicted == 4 or predicted == 6 or predicted == 7 or predicted == 8:
    note_symbols_pos.append(symbolPos)

#Kimenet generálás
canny = cv2.Canny(gray, 75, 200)

lineDistance = round((permaLines[0][1][1] - permaLines[0][0][1]))
print(lineDistance)

detected_circles = cv2.HoughCircles(canny,
                                        cv2.HOUGH_GRADIENT, 1, lineDistance-4, param1=100,
                                        param2=12, minRadius=round(lineDistance/2 - 4), maxRadius=round(lineDistance/2 + 5))
circle_centers = []

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        for pos in note_symbols_pos:
            if pos[0] <= a and pos[0]+pos[2] >= a and pos[1] <= b and pos[1]+pos[3] >= b:
                # cv2.circle(horizontalRGB, (a, b), r, (0, 255, 0), 2)
                circle_centers.append([a,b])
                break

permaNotes = []
for i in range(len(permaLines)):
    permaNotes.append([])

for circle in circle_centers:
    i = 0
    found = False
    while i < len(permaLines) and not found:
        j = 0
        while j < len(permaLines[i]) and not found:
            line=permaLines[i][j]
            cldist=abs(line[1]-circle[1])
            if cldist <= round(lineDistance*3/4):
                if cldist<round(lineDistance/4):
                    permaNotes[i].append([ circle[0], (j+1)*2])
                elif circle[1] < line[1]:
                    permaNotes[i].append([ circle[0], (j+1)*2-1])
                else:
                    permaNotes[i].append([ circle[0], (j+1)*2+1])
                cv2.circle(horizontalRGB, (circle[0], circle[1]), 10, (0, 255, 0), 2)
                found = True 
            j+=1
        i+=1


for row in permaNotes:
    row.sort(key=lambda note: note[0])

finalNotes = []
for i in range(len(permaNotes)):
    finalNotes.append([])

for i in range(len(permaNotes)):
    if row != []:
        for note in permaNotes[i]:
            finalNotes[i].append(note[1])   

output_path='outputs/song'

create_midi(output_path, finalNotes)

    


cv2.imwrite("lines.jpg", lines)
cv2.imwrite("contours.jpg", horizontalRGB)
cv2.imwrite("linesremoved.jpg", horizontal)

cv2.waitKey(0)