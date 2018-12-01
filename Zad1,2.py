import skimage as skimg
from skimage import io
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from scipy import ndimage as ndi
from skimage import feature

img = cv2.imread('gears.bmp')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
czysty = np.ones((300,300))

ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(czysty, contours, -1, (0,255,0), 1)

cv2.imshow('kontury' , czysty)

edges1 = feature.canny(imgray)
edges3 = feature.canny(imgray, sigma=3)
edges2 = feature.canny(imgray, sigma=2)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

cv2.imshow('orginal', img)

ax1.imshow(edges1, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax2.imshow(edges2, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=2$', fontsize=20)

ax3.imshow(edges3, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()
