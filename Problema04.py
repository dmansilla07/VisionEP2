import cv2
import numpy as np
import siamxt

from matplotlib import pyplot as plt

def removeFiosCabelo(img):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return erosion

def filtroLetras(img):
    gray = img.max() - img
    Bc = np.ones((3,3),dtype = bool)
    mxt = siamxt.MaxTreeAlpha(gray, Bc)

    Wmin, Wmax = 10 , 50
    Hmin, Hmax = 10 ,50
    rr = 0.20

    dy = mxt.node_array[7,:] - mxt.node_array[6,:]
    dx = mxt.node_array[10,:] - mxt.node_array[9,:]
    area = mxt.node_array[3,:]
    RR = 1.0*area/(dx*dy)
    
    height = mxt.computeHeight()
    gray_var = mxt.computeNodeGrayVar()
    
    nodes = (dy > Hmin) & (dy < Hmax) & (dx > Wmin) & (dx < Wmax) & (RR > rr) & (gray_var < 15**2) & (dy*2.0 > dx) & (dx*4.0 > dy) & (height > 10)

    #Filtering
    mxt.contractDR(nodes)

    imgFiltered = mxt.getImage()
    return imgFiltered

img = cv2.imread('revista_fapesp.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgSemCab = removeFiosCabelo(img)
imgFiltered = filtroLetras(imgSemCab)

plt.subplot(121),plt.imshow(imgSemCab, cmap = 'gray')
plt.title('Image Sem Cabelos'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(imgFiltered,  cmap = 'gray')
plt.title('Filtered'), plt.xticks([]), plt.yticks([])
plt.show()

