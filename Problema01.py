import cv2
import numpy as np
from matplotlib import pyplot as plt

def dist(u, v, N, M):
    return 1.0*((1.0*u-1.0*N/2.0)**2 + (1.0*v-1.0*M/2)**2)**(1.0/2.0)

def  gaussianMatrix(row, col, d0):
    H = np.ndarray(shape=(row, col), dtype=float)
    for i in range(row):
        for j in range(col):
            H[i][j] = np.exp(-((dist(i, j, row, col)**2)/(2*(d0**2))))
    return H

"""
def validCenter(img, mat, x, y):
    r = mat.shape[0]:

def convolve(img, mat):
    for x in range(img.shape[0]):
        for y in range(img. shape[1]):
            if (valid(img, mat, x, y)):
                for u in range(mat.shape[0]):
                    for v in range(mat.shape[1]):
                        
    return img
"""
img = cv2.imread('leopard_noise.png', 0)
fimg = np.fft.fft2(img)
ffimg = np.fft.fftshift(fimg)
ms = 20*np.log(np.abs(ffimg))
ms1 = np.log(np.abs(ffimg))


rows, cols = img.shape

H = gaussianMatrix(rows, cols, 30)
ffimgr2 = np.multiply(ffimg, H)

print(H)

f_ishift2 = np.fft.ifftshift(ffimgr2)
img_back2 = np.fft.ifft2(f_ishift2)
img_back2 = np.abs(img_back2)

H = gaussianMatrix(rows, cols, 25)
ffimgr = np.multiply(ffimg, H)
f_ishift = np.fft.ifftshift(ffimgr)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after Gaussian Filter 1'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back2, cmap = 'gray')
plt.title('Image after Gaussian Filter 2'), plt.xticks([]), plt.yticks([])

plt.show()
