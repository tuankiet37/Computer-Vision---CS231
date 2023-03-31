import cv2
import numpy as np
card=cv2.imread('9-ro.jpeg')
filter=cv2.imread('template.png')

def correlation(arr1,arr2):
    corr = []
    for i in range(len(arr1)-len(arr2)+1):
        temp = []
        for j in range(len(arr1[0])-len(arr2)+1):
            arr3 = arr1[i:i+len(arr2),j:j+len(arr2[0])] * arr2
            temp.append(np.sum(arr3))
        corr.append(temp)
    return(corr)


h,w,t = filter.shape
arr1=np.array(card)
arr2=np.array(filter)/255.0

temp=correlation(arr1,arr2)
temp = np.array(temp) 
temp = (temp-temp.min()) / (temp.max()-temp.min()) 


min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp)
top_left1 = min_loc
bottom_right1 = (top_left1[0]+w, top_left1[1]+h)
color = (255, 0, 0)

#cv2.circle(card,top_left1,bottom_right1,20, color,5)
cv2.rectangle(card,top_left1,bottom_right1,color,2)
cv2.imshow('card',card)
cv2.waitKey(0)
