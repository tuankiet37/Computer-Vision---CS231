import cv2
import numpy as np

bear = cv2.imread("bear.jpeg")
tiger = cv2.imread("tiger.jpg")
tiger = cv2.resize(tiger, (bear.shape[1], bear.shape[0]))
T=tiger.copy()


for f in range(24):
    t=(f/24)
    T[:,:,:]=(1-t)*bear[:,:,:]+t*tiger[:,:,:]
    cv2.imshow('result', T)
    cv2.waitKey(1)


