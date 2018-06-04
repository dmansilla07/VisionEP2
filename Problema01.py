import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('leopard_noise.png', 0)
fimg = np.fft.fft2(img)
ffimg = np.fft.fftshift(fimg)
ms = 20*np.log(np.abs(ffimg))
ms1 = np.log(np.abs(ffimg))

rows, cols = img.shape
crow,ccol = rows//2 , cols//2
ffimg[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(ffimg)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()
