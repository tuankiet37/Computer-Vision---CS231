import cv2

old_img=cv2.imread('beach.jpg')
new_img=cv2.imread('sky.jpg')
new_img = cv2.resize(new_img, (old_img.shape[1], old_img.shape[0]))
cur_img=old_img.copy()

for delta in range(0,old_img.shape[1],10):
    old_img[:,old_img.shape[1]-delta:,:]=new_img[:,0:delta,:]
    cv2.imshow('cover', old_img)
    cv2.waitKey(1)