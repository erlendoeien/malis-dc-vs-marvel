# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:06:54 2021

@author: 33650
"""
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def reshape_to_uniform_dim(images):
    for i in range(len(images)):
        images[i] = np.reshape(images[i], (images[0].shape[0],images[0].shape[1],3))  
    return images

# image : image we want to crop
# tlc : top left corner, coordinates of the top left corner of the crop
# d : dimension of the square crop
# i : the index allowing us to differentiate all samples that we save
# in imwrite we can specify the folder were we want to save all the data 
def crop_from_tlc(image,tlc,d,i):
    im = image[tlc[0]:tlc[0]+d-1,tlc[1]:tlc[1]+d-1]
    im = np.uint8(im)
    cv2.imwrite('DC_proc/sample_%i.png'%(i), im)

# images : the list of all images we want to crop into samples
# n : the number of crops we do for a given image
# x : the dimansion of the square crop
def crop_images_to_sample_size(images,d,n):
    l = 0
    for i in range(len(images)):
        shape = images[i].shape
        tlc = [0,0]
        for k in range(0,n):
            rand_x = np.uint16(random.uniform(0, shape[1] - d)) # I make sure here that the top left corner cannot be 
            rand_y = np.uint16(random.uniform(0, shape[0] - d)) # initialized in the zone where the crop would step out 
            tlc[0] = rand_y 
            tlc[1] = rand_x
            crop_from_tlc(images[i],tlc,d,l)            
            l += 1

imgs = load_images_from_folder("Marvel/immortal-hulk-2018-1")
#reshape to a uniforme size then cut every one in 6
crop_images_to_sample_size(imgs[1:2], 1000, 10)