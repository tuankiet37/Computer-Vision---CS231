import cv2
import numpy as np


img = cv2.imread('9-ro.jpeg',0)
template = cv2.imread('template.png',0)

h,w = template.shape
img = np.array(img) 
template = np.array(template) / 255


corr = list()
for i in range(img.shape[0]-template.shape[0]+1):
    temp = list()
    for j in range(img.shape[1]-template.shape[1]+1):
        a = img[i:i+template.shape[0],j:j+template.shape[1]] * template
        temp.append(np.sum(a))
    corr.append(temp)

corr = np.array(corr)
corr = (corr-corr.min()) / (corr.max()-corr.min()) 

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
top_left1 = min_loc

bottom_right1 = (top_left1[0]+w, top_left1[1]+h)
cv2.rectangle(img,top_left1,bottom_right1,1)
cv2.imshow('Img',img)
cv2.waitKey(0)