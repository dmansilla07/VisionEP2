import cv2
import numpy as np
import siamxt
from matplotlib import pyplot as plt

img = cv2.imread('knee.pgm', 0)

kernel = np.ones((3, 3), np.uint8)
kernel[0][0] = kernel[2][0] = 0
kernel[1][1] = kernel[0][2] = kernel[2][2] = 0

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

Bc = np.ones((3,3),dtype = bool)

mxt = siamxt.MaxTreeAlpha(neg_img,Bc)

print(kernel)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gradient, cmap = 'gray')
plt.title('Gradient'), plt.xticks([]), plt.yticks([])
plt.show()
