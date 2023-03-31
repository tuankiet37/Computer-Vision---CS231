import cv2

#Cau 1a
img = cv2.imread("lena.jpg")
print("Size of image: ", img.shape()) 

#Cau 1b
dx = img[1:, :] - img[:-1,:]
dy = img[:, 1:] - img[:, :-1]
dx = np.append(dx, img[-1:, :], axis = 0)
dy = np.append(dy, img[:, -1].reshape(-1, 1), axis = 1)