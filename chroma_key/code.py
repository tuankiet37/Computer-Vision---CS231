import cv2
import numpy as np
import math

def find(img,a,alpha):
    H=np.unique(img[:,:,a].flatten())
    return H.max()+alpha, H.min()-(alpha//4)

img=cv2.imread("green.jpg")
print(img.shape)
crop=img[:400,:300,:]

Rmax,Rmin=find(crop,2,20)
Gmax,Gmin=find(crop,1,20)
Bmax,Bmin=find(crop,0,20)

remove=img.copy()
for i in range(remove.shape[0]):
    for j in range(remove.shape[1]):
        if (remove[i,j,0]>=Bmin and remove[i,j,0]<=Bmax):
            if (remove[i,j,1]>=Gmin and remove[i,j,1]<=Gmax):
                if (remove[i,j,2]>=Rmin and remove[i,j,2]<=Rmax):
                    remove[i,j]=0

cv2.imshow("input",img)
cv2.imshow("result",remove)
cv2.waitKey(0)