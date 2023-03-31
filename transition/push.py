import cv2

old_img=cv2.imread('beach.jpg')
new_img=cv2.imread('beach.jpg')

cur_img=old_img.copy()

for delta in range(0, old_img.shape[0],10):
    cur_img[0:old_img.shape[0]-delta,:,:]=old_img[delta:,:,:]
    cur_img[old_img.shape[0]-delta:,:,:]=new_img[0:delta,:,:]
    cv2.imshow('input', cur_img)
    cv2.waitKey(1)