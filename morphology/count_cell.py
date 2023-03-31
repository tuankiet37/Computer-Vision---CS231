import cv2
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX = 0

cell = cv2.imread("blood_cell.png",0)
cell = cv2.bitwise_not(cell) #cell = 255 - cell

kernel = np.ones((4,4),np.uint8)
dilation = cv2.dilate(cell,kernel,iterations = 2)
erosion = cv2.erode(dilation,kernel,iterations = 13)
dilation = cv2.dilate(erosion,kernel,iterations = 4)
img = dilation
ret,thresh = cv2.threshold(img,75,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
a=len(contours)

img =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
img = cv2.drawContours(img, contours, -1, (0,0,255), 3)
cv2.putText(img,'Count: ' + str(a),(0,170), font, 1,(255,0,0),2,cv2.LINE_AA)

cell =  cv2.cvtColor(cell,cv2.COLOR_GRAY2RGB)
cell = cv2.drawContours(cell, contours, -1, (0,0,255), 3)
cv2.imshow("result",img)
cv2.imshow("compare",cell)
cv2.waitKey(0)

