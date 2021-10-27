import cv2
import numpy as np

img = cv2.imread('kepek/tesztkepek/test2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh= cv2.threshold(gray,100,255,cv2.THRESH_BINARY)[1]
thresh = 255-thresh
thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
minLineLength = img.shape[1]/4*3
maxLineGap = 10
lines = cv2.HoughLinesP(thresh,1,np.pi/2,1000,minLineLength,maxLineGap)
goodLines = []
for line in lines:
    x1,y1,x2,y2 = line[0]
    if abs(x2-x1) > abs(y2-y1):
        goodLines.append(line[0])
        
      
goodLines.sort(key=lambda line: line[1])

goodLines2 = []
tempLines = []
i=0
counter = 0
averageGap = 0
while i<len(goodLines):
    tempLines.append(goodLines[i])
    beforeY = goodLines[i][1]
    j=i+1
    while j<len(goodLines) and goodLines[j][1]-beforeY <= 5:
        j += 1
    
   
    i=j+1
    if len(tempLines) == 5:
        goodLines2.append(tempLines)
        averageGap += tempLines[2][1]- tempLines[1][1]
        tempLines = []
        
averageGap = averageGap/len(goodLines2)


for lines in goodLines2:
    for line in lines:
        cv2.line(img,(line[0],line[1]), (line[2],line[3]), (255,255,255), round(averageGap/5))
        cv2.line(thresh,(line[0],line[1]), (line[2],line[3]), (0,0,0), round(averageGap/5))

for lines in goodLines2:
    print(lines)
print(len(goodLines2))

thresh = cv2.erode(thresh,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=3)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)

    cv2.drawContours(img, contours, -1, (255,0,255), thickness=2)

cv2.imwrite('linetest.jpg',img)
cv2.imwrite('linethresh.jpg',thresh)
cv2.waitKey(0)