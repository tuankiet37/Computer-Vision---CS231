import cv2
import numpy as np
import math
import random
from scipy.ndimage import convolve
from tqdm import trange

def energy(img):
    dx = img[1:, :] - img[:-1,:]
    dy = img[:, 1:] - img[:, :-1]
    dx = np.append(dx, img[-1:, :], axis = 0)
    dy = np.append(dy, img[:, -1].reshape(-1, 1), axis = 1)

    f = np.array(np.sqrt(dx**2 + dy**2), dtype = 'float32')
    f = 255 * (f-f.min())/(f.max()-f.min())
    return f

def calc_energy(img):
    f1 = energy(img[:,:,0])
    f2 = energy(img[:,:,1])
    f3 = energy(img[:,:,2])
    f=f1+f2+f3
    return f

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=int)
    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=np.bool_)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))

    return img


def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c): # use range if you don't want to use tqdm
        img = carve_column(img)

    return img

scale = 0.5
img = cv2.imread('beach.jpg')
out = crop_c(img, scale)
cv2.imshow('Orginal', img)
cv2.imshow('Crop', out)
cv2.waitKey(0)
 

