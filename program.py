#Könyvtárak beimportálása
import cv2
import numpy as np

#Kép beolvasása és transzformálása
img = cv2.imread('kepek/tesztkepek/test2.jpg')
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
vertical = np.copy(thresh)
rows = vertical.shape[0]
verticalSize = 5
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)

vertical = cv2.bitwise_not(vertical)
edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel)

smooth = np.copy(vertical)

smooth = cv2.blur(smooth, (2, 2))

(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]

#Alakzatok kontúrjainak megkeresése és megjelölése
vertical_reverse = 255-vertical
contours, hierarchy = cv2.findContours(vertical_reverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

verticalRGB = cv2.cvtColor(vertical,cv2.COLOR_GRAY2BGR)
cv2.drawContours(verticalRGB, contours, -1, (0,0,255), thickness=2)

#Alakzatok eltárolása
symbols = []

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)

    cv2.rectangle(verticalRGB, (x,y), (x+w, y+h), (255,0,255), thickness=2)

    symbols.append(vertical[y:y+h, x:x+w])

symbols = sorted(symbols, key=lambda symbol: symbol.shape[1], reverse=True)

#c = 1
#while c <= 575:
#    cv2.imwrite("template"+str(c)+".png", symbols[c])
#    c = c + 1

#Template matching a hangjegyek megtalálására
notes = []
#thresholds = [0.7, 0.834, 0.7935, 0.75, 0.83, 0.8, 0.75, 0.85, 0.75, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.7, 0.72, 0.75, 0.815, 0.7715, 0.7, 0.741, 0.75, 0.675, 0.85, 0.68]
#templates = [cv2.imread('kepek/template'+str(c)+'.jpg',cv2.IMREAD_GRAYSCALE) for c in range(1,30)]
c = 1
while c <= 3:
    template = cv2.imread('kepek/ujtemplatek/template'+str(c)+'.png')
    temp_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    temp_thresh = cv2.threshold(temp_gray,100,255,cv2.THRESH_BINARY)[1]
    temp_thresh = 255 - temp_thresh
    w = template.shape[1]
    h = template.shape[0]
    result = cv2.matchTemplate(thresh, temp_thresh, cv2.TM_CCOEFF_NORMED)
    threshold = 0.57
    loc = np.where( result >= threshold)
    for pt in zip(*loc[::-1]): 
        appendable = True
        for i in notes:
            if abs(i[0][0]-pt[0])<10 and abs(i[0][1]-pt[1])<10:
                appendable = False
                break
        if appendable:
            notes.append([pt, c])
            cv2.rectangle(thresh, pt, (pt[0] + w, pt[1] + h), (0,0,0), -1)
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    c=c+1

#template = cv2.imread("kepek/templatek/template1.jpg")
#temp_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
#w = template.shape[1]
#h = template.shape[0]
#result = cv2.matchTemplate(gray, temp_gray, cv2.TM_CCOEFF_NORMED)
#threshold = thresholds[c-1]
#loc = np.where( result >= threshold)
#for pt in zip(*loc[::-1]):
#    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite("tempmatch.jpg", img)
cv2.imwrite("tempremoved.jpg", gray)

#cv2.imshow("symbol", symbols[141])

cv2.imwrite("contours.jpg", verticalRGB)
cv2.imwrite("linesremoved.jpg", vertical)
cv2.waitKey(0)