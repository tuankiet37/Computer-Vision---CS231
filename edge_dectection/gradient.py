import cv2
import numpy as np
import math

img = cv2.imread('annya.jpg', 0)
h, w = img.shape

dx = img[1:, :] - img[:-1,:]
print(dx.shape)
dy = img[:, 1:] - img[:, :-1]
print(dy.shape)
dx = np.append(dx, img[-1:, :], axis = 0)
print(dx.shape)
dy = np.append(dy, img[:, -1].reshape(-1, 1), axis = 1)
print(dy.shape)

f = np.array(np.sqrt(dx**2 + dy**2))
f = ((f-f.min())/(f.max()-f.min()))*255
f = f.astype('uint8')

cv2.imshow("Result",f)
cv2.waitKey(0)
