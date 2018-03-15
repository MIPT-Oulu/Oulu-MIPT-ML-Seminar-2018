"""
Data augmentations.

(c) Aleksei Tiulpin, University of Oulu, 2018

"""


import numpy as np
import cv2
import pandas as pd
import random
from io import BytesIO

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def augment_random_flip(img, hprob=0.5, vprob=0.5):
    
    img = img.copy()
    if random.random() > hprob:
        img = cv2.flip(img, 1)

    if random.random() > vprob:
        img = cv2.flip(img, 0)
        
    return img

def augment_random_crop(x, crop_size):
    
    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)
    
    h, w = x.shape[0], x.shape[1]

    if w < crop_size[0] or h < crop_size[1]:
        x = x[:]
        pad_w = np.uint16((crop_size[0] - w)/2) + int(crop_size[0]*.1)
        pad_h = np.uint16((crop_size[1] - h)/2) + int(crop_size[1]*.1)
        x = cv2.copyMakeBorder(x,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=0)
        h, w = x.shape[0], x.shape[1]
    
    x1 = random.randint(0, w - crop_size[0])
    y1 = random.randint(0, h - crop_size[1])
    return x[y1:y1+crop_size[1], x1:x1+crop_size[0],:]


def center_crop(x, crop_size):
    
    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)
    
    h, w = x.shape[0], x.shape[1]

    if w < crop_size[0] or h < crop_size[1]:
        x = x[:]
        pad_w = np.uint16((crop_size[0] - w)/2) + int(crop_size[0]*.1)
        pad_h = np.uint16((crop_size[1] - h)/2) + int(crop_size[1]*.1)
        x = cv2.copyMakeBorder(x,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=0)
        h, w = x.shape[0], x.shape[1]

    x1 = w//2-crop_size[0]//2
    y1 = h//2-crop_size[1]//2
    img_pad = x[y1:y1+crop_size[1], x1:x1+crop_size[0],:]
    return img_pad

def augment_random_linear(img, sr=5, ssx=0.1, ssy=0.1, inter=cv2.INTER_LINEAR):

    rot = (np.random.rand(1)[0]*2-1)*sr
    scalex = np.random.rand(1)[0]*ssx
    scaley = np.random.rand(1)[0]*ssy
    
    R = np.array([np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot)), 0, 
                  -np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0,
                  0, 0, 1
                 ]).reshape((3,3))
    
    S = np.array([1, scalex, 0, 
                  scaley, 1, 0,
                 0, 0, 1]).reshape((3,3))
    
    A = np.dot(R, S)

    return cv2.warpAffine(img, A.T[:2, :], img.shape[1::-1], inter, borderMode=cv2.BORDER_REFLECT)