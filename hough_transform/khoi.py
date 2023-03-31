import cv2
import numpy as np

def energy(img):
    dx = img[1:, :] - img[:-1,:]
    dy = img[:, 1:] - img[:, :-1]
    dx = np.append(dx, img[-1:, :], axis = 0)
    dy = np.append(dy, img[:, -1].reshape(-1, 1), axis = 1)

    f = np.array(np.sqrt(dx**2 + dy**2), dtype = 'float32')
    f = 255 * (f-f.min())/(f.max()-f.min())
    f = np.uint8(f)
    return f

# Step 1: accumulate hough space
def hough_transform(edge):
    h, w = edge.shape
    diag = int(np.ceil(np.sqrt(w * w + h * h)))
    diag_step = (2 * diag)
    theta_range = np.deg2rad(np.arange(0, 360, 1))
    pros = np.linspace(-diag, diag,diag_step)
    #theta_range = np.arange(-3.14, 3.14, 0.01)
    sin_thetas = np.sin(theta_range)
    cos_thetas = np.cos(theta_range)
    H = np.zeros((len(pros), len(theta_range)),dtype='uint8')
    
    edge_points = np.where(edge > 150)
    x_cos = np.dot(edge_points[1].reshape((-1,1)),cos_thetas.reshape((1,-1)))
    y_sin = np.dot(edge_points[0].reshape((-1,1)),sin_thetas.reshape((1,-1)))
    accumulator = (x_cos + y_sin).astype('int64')

    for i in range(len(theta_range)):
        pro,counts = np.unique(accumulator[:,i], return_counts=True)
        H[pro,i] = counts
        
    return H

img = cv2.imread('pitch.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Step 0: Calculate the energy of img
edge = energy(gray)

#transform to hough space
temp = img.copy()
H = hough_transform(edge)

H_points = np.where(H > 95)
for pros,theta in zip(H_points[0],H_points[1]):
    a = np.cos(np.deg2rad(theta))
    b = np.sin(np.deg2rad(theta))
    x0 = a*pros
    y0 = b*pros
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))
    cv2.line(temp, (x1,y1), (x2,y2), (0,0,255), 3, cv2.LINE_AA)
    
cv2.imshow('img',temp)
cv2.waitKey(0)
