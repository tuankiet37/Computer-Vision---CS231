import cv2

old_img=cv2.imread('beach.jpg')
new_img=cv2.imread('sky.jpg')
new_img = cv2.resize(new_img, (old_img.shape[1], old_img.shape[0]))
cur_img=old_img.copy()

for delta in range(0,old_img.shape[1],10):
    cur_img[:,:old_img.shape[1]-delta,:]=old_img[:,delta:,:]
    cur_img[:,old_img.shape[1]-delta:,:]=new_img[:,old_img.shape[1]-delta:,:]
    cv2.imshow('uncover', cur_img)
    cv2.waitKey(1)