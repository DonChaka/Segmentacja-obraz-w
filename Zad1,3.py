import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2
from skimage import feature
from skimage.feature import peak_local_max
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import quickshift
from skimage.segmentation import felzenszwalb
from skimage.morphology import watershed
from skimage.feature import peak_local_max

img = cv2.imread('fish.bmp')
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

distance = ndi.distance_transform_edt(im)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=im)
markers = ndi.label(local_maxi)[0]
ret1 = watershed(-distance, markers, mask = im)
ryba1 = mark_boundaries(im, ret1)
cv2.imshow('watershed', ryba1)

ret2 = slic(img, n_segments = 100, compactness = 20)
ryba2 = mark_boundaries(img, ret2)
cv2.imshow('slic', ryba2)

ret3 = quickshift(img)
ryba3 = mark_boundaries(img, ret3)
cv2.imshow('quickshift', ryba3)

ret4 = felzenszwalb(img, 300.0, sigma = 0.95, min_size = 150)
ryba4 = mark_boundaries(img, ret4)
cv2.imshow('falzenszwalb', ryba4)
