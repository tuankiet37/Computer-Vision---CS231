import cv2
import numpy as np
def correlation(arr1,arr2):
    corr = []
    for i in range(len(arr1)-len(arr2)+1):
        temp = []
        for j in range(len(arr1[0])-len(arr2)+1):
            arr3 = arr1[i:i+len(arr2),j:j+len(arr2[0])] * arr2
            temp.append(np.sum(arr3))
        corr.append(temp)
    print(corr)

arr1 = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12]])

arr2 = np.array([[0,1],
                [1,0]])

correlation(arr1,arr2)