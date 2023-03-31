import cv2

fg=cv2.imread('hi.jpg')
eff = cv2.imread('snow.jpg')
mask = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)

fg = cv2.resize(fg, (mask.shape[1], mask.shape[0]))
eff = cv2.resize(eff, (mask.shape[1], mask.shape[0]))

result = fg.copy()
alpha = 0.5
result[mask[:,:,3] != 0] = fg[mask[:,:,3] != 0] * alpha + eff[mask[:,:,3] != 0] * (1 - alpha)

cv2.imshow('Result', result)
cv2.waitKey(0)