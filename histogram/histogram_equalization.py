import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("low.png",0)
img=np.array(img)
flat = img.flatten()

#hist,bin = np.histogram(img.ravel(),bins = 256)
def count(img, size):
    histogram = np.zeros(size)
    for pixel in img:
        histogram[pixel] += 1
    return histogram
    
print(img)
hist = count(img,img.shape[0])
cdf = np.cumsum(hist)

nj = (cdf - cdf.min())
N = cdf.max() - cdf.min()
cdf = (nj /N)*255
cdf = cdf.astype('uint8')

img_new = cdf[flat]
print(cdf[0])


# img_new = np.reshape(img_new, img.shape)
# cv2.imshow('input',img)
# cv2.imshow('result',img_new)
# cv2.waitKey(0)