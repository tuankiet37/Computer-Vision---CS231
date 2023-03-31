import cv2

old_img=cv2.imread('beach.jpg')
#new_img=cv2.imread('beach.jpg')

new_img = old_img[:,::-1,:]
cv2.imshow('input', old_img)
cv2.imshow('flip', new_img)
cv2.waitKey(0)