import cv2
import numpy as np
import math

img = np.zeros((800, 800, 3), dtype = "uint8")
img.fill(255)
mp=int(800*math.sqrt(3)/2)
a = np.array([400, 0])
b = np.array([0, mp])
c = np.array([800, mp])

def triangle(a,b,c,n):
    if (n==0):
        return
    cv2.line(img, a, b, (107, 124, 238), 2)
    cv2.line(img, b, c, (107, 124, 238), 2)
    cv2.line(img, a, c, (107, 124, 238), 2)
    triangle((a+b)//2, b, (b+c)//2, n-1)
    triangle(a, (a+b)//2, (a+c)//2, n-1)
    triangle((a+c)//2, c, (b+c)//2, n-1)

n=6
triangle(a,b,c,n)
cv2.imshow('fractal triangle', img) 
cv2.waitKey()