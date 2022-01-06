# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:06:54 2021

@author: 33650
"""
import numpy as np
import random
import cv2
import os
from pathlib import Path


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(folder, filename)
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def load_all_images(img_category_folder):
    imgs_flattened = []
    for comic_folder_path, subdirs, pages in os.walk(img_category_folder):
        print(comic_folder_path)
        imgs_flattened.extend(
            [
                cv2.imread(os.path.join(comic_folder_path, page))
                for page in pages
                if page.endswith(".jpg")
            ]
        )
    return imgs_flattened


def reshape_to_uniform_dim(images):
    for i in range(len(images)):
        images[i] = np.reshape(images[i], (images[0].shape[0], images[0].shape[1], 3))
    return images


# image : image we want to crop
# tlc : top left corner, coordinates of the top left corner of the crop
# d : dimension of the square crop
# i : the index allowing us to differentiate all samples that we save
# in imwrite we can specify the folder were we want to save all the data
def crop_from_tlc(image, tlc, d, i, output_dir):
    im = image[tlc[0] : tlc[0] + d - 1, tlc[1] : tlc[1] + d - 1]
    if im.size != 0:
        im = np.uint8(im)
        cv2.imwrite(os.path.join(output_dir, f"sample_{str(i)}.png"), im)

def crop_image_by_width(image, res):
    """
    Crops the image into as many samples as possible with 
    resolution res x res (quadratic)

    image: ndarray
    res: integer
    """
    samples = []
    M, N, _ = image.shape
    tlc = [0, 0]
    y, x = res, res
    for _row in range(0, (M//res)): # Could be optimized so crops are centered, based on remainder
        for _col in range(1, N//res + 1):
            sample = image[tlc[0]:y, tlc[1]:x]
            samples.append(sample)
            # Moving seeker horizontally
            tlc[1]=x
            x +=res
        # Resetting seeker to start at next row
        x = res
        tlc=[y, 0]
        y += res
    return np.array(samples, dtype="float")



    


# images : the list of all images we want to crop into samples
# n : the number of crops we do for a given image
# x : the dimansion of the square crop
def crop_images_to_sample_size(images, d, n, output_dir):
    l = 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(len(images)):
        shape = images[i].shape
        tlc = [0, 0]
        for k in range(0, n):
            rand_x = np.uint16(
                random.uniform(0, shape[1] - d)
            )  # I make sure here that the top left corner cannot be
            rand_y = np.uint16(
                random.uniform(0, shape[0] - d)
            )  # initialized in the zone where the crop would step out
            tlc[0] = rand_y
            tlc[1] = rand_x
            crop_from_tlc(images[i], tlc, d, l, output_dir)
            l += 1


if __name__ == "__main__":
    data_path = os.path.realpath("data")
    # Read all images and flatten them - Including advertisement/covers
    marvel_imgs = load_all_images(os.path.join(data_path, "Marvel"))
    dc_imgs = load_all_images(os.path.join(data_path, "DC"))

    processed_path = os.path.join(data_path, "processed", "crop_6_1000x1000")
    # Crop all images into 6 samples given the page resolution of 3056x1988, each of resolution 1000x1000 (-1 so in fact 999x999)
    # To minimize overlap between samples
    crop_images_to_sample_size(marvel_imgs, 1000, 6, os.path.join(processed_path, "marvel"))
    crop_images_to_sample_size(dc_imgs, 1000, 6, os.path.join(processed_path, "dc"))
