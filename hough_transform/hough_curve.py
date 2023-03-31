import cv2
import numpy as np

def deg2grad(deg):
    return deg*3.141592654/180
# Step 0.5: initialize hough table
theta_range = np.arange(-3.14, 3.14, 0.01)
H = np.zeros((500, len(theta_range)), dtype=np.uint8)
# Step 1: accumulate hough space
def accumulate(point):
    #for theta in range(360):
    for theta in theta_range:
        pro = point[0]*np.cos(theta) + point[1]*np.sin(theta)
        # If pro in range of Hough space
        if pro >= 0 and pro < 500:
            # map theta to Hough space
            # (theta - (-3.14))/0.01
            H[int(pro), int((theta+3.14)/0.01)] += 250
    return H

accumulate([70.71,70.71])
accumulate([0, 141.42])
accumulate([141.42,0])

# accumulate([0, 0])
# accumulate([100, 100])
# accumulate([200, 200])
# accumulate([300, 300])
# accumulate([400, 400])

cv2.imshow("Hough space visualization", H)
cv2.waitKey(0)
