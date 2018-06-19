import cv2
import numpy as np
import siamxt

from matplotlib import pyplot as plt

def removeFiosCabelo(img):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return erosion

def filtroLetras(img):
    gray = img
    Bc = np.ones((3,3),dtype = bool)
    mxt = siamxt.MaxTreeAlpha(gray, Bc)
    V = mxt.computeVolume()
    ext = mxt.computeExtinctionValues(V, "volume")
    mxt.extinctionFilter(ext, 10)
    imgFiltered = mxt.getImage()
    return imgFiltered

img = cv2.imread('revista_fapesp.png', 0)
img = removeFiosCabelo(img)
imgFiltered = filtroLetras(img)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(imgFiltered,  cmap = 'gray')
plt.title('Filtered'), plt.xticks([]), plt.yticks([])
plt.show()

