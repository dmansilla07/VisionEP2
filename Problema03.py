import cv2
import numpy as np
import siamxt

from matplotlib import pyplot as plt

dx = [1, 1, -1, -1, 0, 1, 0, -1]
dy = [1, -1, 1, -1, 1, 0, -1, 0]

def valid(i, j, rows, cols):
    if (i>=0 and j>=0 and i<rows and j<cols):
        return 1
    else:
        return 0

def markToC(x, maxi, mini):
    return (1.0*x-mini)*(256.0/(maxi-mini))

def colorFunction(img, markers):
    img_final = img.copy()
    rows, cols = img.shape
    maxi = markers.max()
    mini = markers.min()
    for i in range(0, rows):
        for j in range(0, cols):
            img_final[i][j] = markToC(markers[i][j], maxi, mini)
    return img_final

imgW = cv2.imread('knee.pgm')
img = cv2.imread('knee.pgm', 0)
img_max = img.max()
neg_img = img_max - img

kernel = np.ones((3, 3), np.uint8)
kernel[0][0] = kernel[2][0] = 0
kernel[1][1] = kernel[0][2] = kernel[2][2] = 0

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
ImgMax = gradient.max()
gray = ImgMax - gradient

Bc = np.ones((3,3),dtype = bool)

mxt = siamxt.MaxTreeAlpha(gray, Bc)

V = mxt.computeVolume()
H = mxt.computeHeight()
Vext = mxt.computeExtinctionValues(V, "volume")
Hext = mxt.computeExtinctionValues(H, "height")

mxt.extinctionFilter(Vext, 8)
#mxt.extinctioxnFilter(Hext, 8)

img_filtered = ImgMax - mxt.getImage()
#img_filtered = (img_filtered)

ret, thresh = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('a', thresh)
cv2.waitKey(0)


ret, markers = cv2.connectedComponents(thresh)

img2 = colorFunction(img_filtered, markers)
cv2.imshow('a', img2)
cv2.waitKey(0)

markers = cv2.watershed(imgW,markers)
imgW[markers == -1] = [255,0,0]
cv2.imshow('a', imgW)
cv2.waitKey(0)
