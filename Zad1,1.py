import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('brain_tumor.jpg',0)

ret1, th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)

th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

ret4, th4 = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

ret,nath = cv.threshold(img, 237, 255, cv.THRESH_BINARY)

cv.imshow('Orginal' , img)
cv.imshow('adaptive Threshold Binary', th1)
cv.imshow('Thresh OTSU', th4)
cv.imshow('adaptive Threshold Mean', th2)
cv.imshow('adaptive Threshold Gaussian', th3)
cv.imshow('Wartosc progowa: 237', nath)
ret,nath = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
cv.imshow('Wartosc progowa: 200', nath)
ret,nath = cv.threshold(img, 190, 255, cv.THRESH_BINARY)
cv.imshow('Wartosc progowa: 190', nath)
