import cv2
import numpy as np
import siamxt
from matplotlib import pyplot as plt

def imgSemBuracos(img, area):
    Bc = np.ones((3,3),dtype = bool)
    maxi = img.max()
    gray = maxi - img
    mxt = siamxt.MaxTreeAlpha(gray, Bc)
    mxt.areaOpen(area)
    imgSemBuracos = maxi - mxt.getImage()
    return imgSemBuracos


img = cv2.imread('fruit.png', 0)
imgFiltered = imgSemBuracos(img, 50)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(imgFiltered, cmap = 'gray')
plt.title('Image filtered'), plt.xticks([]), plt.yticks([])
plt.show()

