import cv2
import Drawing
import torch
import numpy as np


# interprets the image as a matrix of pixels
def interpreter():

    Drawing.main()
    img = cv2.imread('number.png', 0)
    pixels = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixels.append(255 - img[i][j])
    return torch.tensor(torch.from_numpy(np.asarray(pixels)))
