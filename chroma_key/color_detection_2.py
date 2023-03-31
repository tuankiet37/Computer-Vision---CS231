import cv2 
import numpy as np
from sklearn.linear_model import LogisticRegression

bg = cv2.imread("bg.jpg")
fg = cv2.imread("fg.jpg")
img = cv2.imread("green.jpg")

#reshape 3 channels of back ground
x1 = bg.reshape(-1,3)

#reshape 3 channels of fore ground
x2 = fg.reshape(-1,3)

#reshape 3 channels of predicted img
pred = img.reshape(-1,3)

#concatenate each column
X1 = np.concatenate((x1[:,0],x2[:,0])).reshape(-1,1)
X2 = np.concatenate((x1[:,1],x2[:,1])).reshape(-1,1)
X3 = np.concatenate((x1[:,2],x2[:,2])).reshape(-1,1)

#initialize labels for bg and fg
y1 = np.ones(x1.shape[0])
y2 = np.zeros(x2.shape[0])

#concatenate all
X = np.concatenate((X1,X2,X3),axis=1)
y = np.concatenate((y1,y2))

#use LogisticRegression to predict fg and bg
clf = LogisticRegression(random_state=0).fit(X, y)
pred_label = clf.predict(pred).astype('bool')

#remove bg
pred[pred_label] = 0

#show img
pred = pred.reshape(img.shape)
cv2.imshow('result',pred)
cv2.waitKey(0)