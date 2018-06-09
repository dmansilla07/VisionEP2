import cv2
import numpy as np
from matplotlib import pyplot as plt

def dist(u, v, N, M):
    return 1.0*((1.0*u-1.0*N/2.0)**2 + (1.0*v-1.0*M/2)**2)**(1.0/2.0)

def  gaussianMatrix(row, col, d0):
    H = np.ndarray(shape=(row, col), dtype=float)
    for i in range(row):
        for j in range(col):
            H[i][j] = np.exp(-((dist(i, j, row, col)**2)/(2.0*(d0**2))))
    return H

def gaussianFilter(r, d0):
    sz = 2*r+1
    H = np.ndarray(shape=(sz, sz), dtype=float)
    for i in range(sz):
        for j in range(sz):
            H[i][j] = np.exp(-((dist(i, j, sz, sz)**2)/(2.0*(d0**2))))
    return H

def  butterworthMatrix(row, col, d0):
    H = np.ndarray(shape=(row, col), dtype=float)
    for i in range(row):
        for j in range(col):
            H[i][j] = 1.0/(1.0+(dist(i, j, row, col)/d0)**2)
    return H  

def lowpassFilter(row, col, d0):
    H = np.ndarray(shape=(row,col), dtype=float)
    for i in range(row):
        for j in range(col):
            val = 0
            if (dist(i, j, row, col) <= d0):
                val = 1
            H[i][j] = val
    return H

def validCenter(img, mat, x, y):
    r = mat.shape[0]//2
    N, M = img.shape;
    if ((x-r < 0) or (y-r < 0)):
        return 0;
    if ((x+r >= N) or (y+r >= M)):
        return 0;
    return 1;

def convolution(img, mat):
    r = mat.shape[0]//2
    x = 0
    y = 0
    while (x < img.shape[0]):
        valid1 = 0
        while (y < img.shape[1]):
            valid2 = 0
            if (validCenter(img, mat, x, y) == 1):
                valid2 = 1
                valid1 = 1
                for u in range(mat.shape[0]):
                    for v in range(mat.shape[1]):
                        x_1 = x - (r - u)
                        y_1 = y - (r - v)
                        if (x_1 < 2 and y_1 < 2):
                            print(x_1, " ", y_1)
                        img[x_1][y_1] = img[x_1][y_1] * mat[u][v]
            if (valid2 == 0):
                y = y + 1
            else:
                y = y + 2*r + 1
        y = 0
        if (valid1 == 0):
            x = x + 1
        else:
            x = x + 2*r + 1
    return img


def customShift(img):
    rows, cols = img.shape
    for i in range(0, rows):
        for j in range(0, cols):
            img[i][j] = img[i][j]*((-1.0)**(i+j))
    return img
            

def media(img, mat):
    r = mat.shape[0]//2
    x = 0
    y = 0
    while (x < img.shape[0]):
        valid1 = 0
        while (y < img.shape[1]):
            valid2 = 0
            if (validCenter(img, mat, x, y) == 1):
                valid2 = 1
                valid1 = 1
                l = []
                for u in range(mat.shape[0]):
                    for v in range(mat.shape[1]):
                        x_1 = x - (r - u)
                        y_1 = y - (r - v)
                        l.append(img[x_1][y_1])
                mediana = np.median(l)
                for u in range(mat.shape[0]):
                    for v in range(mat.shape[1]):
                        x_1 = x - (r - u)
                        y_1 = y - (r - v)
                        img[x_1][y_1] = mediana   
            if (valid2 == 0):
                y = y + 1
            else:
                y = y + 2*r + 1
        y = 0
        if (valid1 == 0):
            x = x + 1
        else:
            x = x + 2*r + 1
    return img



img = cv2.imread('leopard_noise.png', 0)
fimg = np.fft.fft2(img)
ffimg = np.fft.fftshift(fimg)
#ffimg = fimg

ms = 20*np.log(np.abs(ffimg))
ms1 = np.log(np.abs(ffimg))

rows, cols = img.shape

H1 = gaussianMatrix(rows, cols, 55)
ffimgr2 = np.multiply(ffimg, H1)


f_ishift2 = np.fft.ifftshift(ffimgr2)
img_back2 = np.fft.ifft2(f_ishift2)
img_back2 = np.abs(img_back2)

H2 = lowpassFilter(rows,cols,70)
H3 = butterworthMatrix(rows,cols, 70)
print(ffimg[0][0])

ffimgr = np.multiply(ffimg, H3)

print(ffimgr[0][0])

f_ishift = np.fft.ifftshift(ffimgr)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after Lowpass'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back2, cmap = 'gray')
plt.title('Image after Gaussian'), plt.xticks([]), plt.yticks([])

plt.show()
