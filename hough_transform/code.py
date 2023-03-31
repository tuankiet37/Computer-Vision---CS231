import cv2  
import numpy as np
import math
from tqdm import trange

img=cv2.imread("pitch.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold_energy = 150
threhold_H = 95

def energy(img):
    dx = img[1:, :] - img[:-1,:]
    dy = img[:, 1:] - img[:, :-1]
    dx = np.append(dx, img[-1:, :], axis = 0)
    dy = np.append(dy, img[:, -1].reshape(-1, 1), axis = 1)
    f = np.array(np.sqrt(dx**2 + dy**2), dtype = 'float32')
    f = 255 * (f-f.min())/(f.max()-f.min())
    f = np.uint8(f)
    return f

def accumulate(t1,t2):
    energy_map = energy(gray)
    h,w = energy_map.shape
    d = int(math.sqrt(h**2 + w**2)) #diagonal line
    theta_range = np.arange(0,180,1)
    
    #Vectorized
    sin = np.sin(np.deg2rad(theta_range))
    cos = np.cos(np.deg2rad(theta_range))
    line_idx = np.where(energy_map > t1) 
    x_cos = np.dot(line_idx[1].reshape((-1,1)),cos.reshape((1,-1)))
    y_sin = np.dot(line_idx[0].reshape((-1,1)),sin.reshape((1,-1)))
    p = (x_cos + y_sin).astype('int')
    H = np.zeros(((2*d)+1, len(theta_range)), dtype=np.uint8)

    for i in trange(len(theta_range)):
        pro,counts = np.unique(p[:,i], return_counts=True)
        H[pro,i] = counts
    H_points = np.where(H >t2)
    return H_points

H_points=accumulate(threshold_energy,threhold_H)
for pros,theta in zip(H_points[0],H_points[1]):
    a = np.cos(np.deg2rad(theta))
    b = np.sin(np.deg2rad(theta))
    x0 = a*pros
    y0 = b*pros
    x1 = int(x0 + 1500*(-b))
    y1 = int(y0 + 1500*(a))
    x2 = int(x0 - 1500*(-b))
    y2 = int(y0 - 1500*(a))
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 3, cv2.LINE_AA)
cv2.imshow('Detect',img)
cv2.waitKey(0) 