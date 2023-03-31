import cv2
img=cv2.imread('download.png')
delta=100
bright_img=cv2.add(img,delta)
cv2.imshow('input', img)
cv2.imshow('test', bright_img)
cv2.waitKey()