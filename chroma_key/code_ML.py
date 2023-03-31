import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import tree
bg=cv2.imread("background.jpg")
fg=cv2.imread("foreground.jpg")
img=cv2.imread("green.jpg")
#Data train
x1 = bg.reshape(-1,3)
x2 = fg.reshape(-1,3)
X=np.concatenate((x1,x2))
#print(X.shape)

y1=np.ones((bg.shape[0]*bg.shape[1]))
y2=np.zeros((fg.shape[0]*fg.shape[1]))
Y=np.concatenate((y1,y2))
#print(Y.shape)

#Data test
X_test = img.reshape(-1,3)
#print(X_test.shape)

# clf = tree.DecisionTreeClassifier(criterion="entropy")
# clf.fit(X,Y)
clf = LogisticRegression().fit(X, Y)
Y_test = clf.predict(X_test).astype('bool')

X_test[Y_test]=0

X_test = X_test.reshape(img.shape)
cv2.imshow('result',X_test)
cv2.waitKey(0)

