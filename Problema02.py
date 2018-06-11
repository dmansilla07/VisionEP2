import cv2
import numpy as np
import siamxt
from matplotlib import pyplot as plt

img = cv2.imread('fruit.png', 0)

Bc = np.ones((3,3),dtype = bool)
 
#Negating the image
img_max = img.max()
neg_img = img_max - img

# Building the max-tree of the negated image, i.e. min-tree
mxt = siamxt.MaxTreeAlpha(neg_img,Bc)

print(mxt)

#Filtering the min-tree
a = 100
mxt.areaOpen(a)
img_filtered = img_max - mxt.getImage()

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_filtered, cmap = 'gray')
plt.title('Image filtered'), plt.xticks([]), plt.yticks([])
plt.show()

