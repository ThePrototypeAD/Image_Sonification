# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:30:06 2025

@author: deand
"""
#%% imports
import cv2 as cv          # adaptive thresholding -> in theory, cv can handle everything that skimage uses, well prolly not the save image procedure
import numpy as np        # array handlings
import skimage.color      # global thresholding
import skimage.filters
import skimage.io
import skimage.transform
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy import sparse as sparse
from scipy.io import wavfile

#importing local project modules
import sys
sys.path.insert(1, r"C:\Users\deand\__Github_clones")
sys.path.insert(1, r'D:\Documents\__Projects')

# import gilbert.gilbert2d as gilbert
from Image_Sonification.Reader import Image_read



#%%
# if __name__ == "__main__":
    
#     im_height = 1080
#     im_width = 1920
    
#     image_hls = np.zeros((im_height, im_width, 3)) 
#     test_hue = np.linspace(0, 180, im_width)
#     for i in range(len(image_hls)):
#         image_hls[i, :, 0] = test_hue[:]
#         image_hls[i, :, 1] = 128
#         image_hls[i, :, 2] = 255
#     image_hls = image_hls.astype('uint8')
    
#     image_rgb = cv.cvtColor(image_hls, cv.COLOR_HLS2RGB) #check the coordinate again tho
    
#     gilbert_x = []
#     gilbert_y = []

#     for x, y in gilbert.gilbert2d(im_width, im_height):
#         gilbert_x.append(x)
#         gilbert_y.append(y)
    
#     # plt.plot(gilbert_x, gilbert_y)
#     # plt.show()
    
#     plt.imshow(image_rgb)
#     # plt.plot(gilbert_x, gilbert_y, alpha=0.4, color='black', label='Gilbert scan')
#     # plt.legend()
#     plt.title('original_image')
#     plt.show()
    
#     dim_1d = len(gilbert_x)
#     gilbert_scan_1d = np.zeros((dim_1d, 3))
    
#     #scalable up to 1920 x 1080 resolution
#     # todo: check scalability for commons
#     # also utilize scipy.sparse for memory manipulation for large images
    
#     gilbert_scan_1d_image = np.zeros((int(dim_1d/300), int(dim_1d/100), 3))  
#     # image form put the scan into the width dimension (x), arbiraty set the width dimesion for clarity
#     # for image form, divide dimension to save memory
    
#     #not scalable for large dimensions!
#     for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
        
#         gilbert_scan_1d[x, 0] = image_rgb[gilbert_y[x], gilbert_x[x], 0]
#         gilbert_scan_1d[x, 1] = image_rgb[gilbert_y[x], gilbert_x[x], 1]
#         gilbert_scan_1d[x, 2] = image_rgb[gilbert_y[x], gilbert_x[x], 2]
        
#         if (x%100 == 0):
#             gilbert_scan_1d_image[:, int(x/100), 0] = image_rgb[gilbert_y[x], gilbert_x[x], 0]
#             gilbert_scan_1d_image[:, int(x/100), 1] = image_rgb[gilbert_y[x], gilbert_x[x], 1]
#             gilbert_scan_1d_image[:, int(x/100), 2] = image_rgb[gilbert_y[x], gilbert_x[x], 2]
    
#     gilbert_scan_1d_image = gilbert_scan_1d_image.astype('uint8')
#     plt.imshow(gilbert_scan_1d_image)
#     plt.title('Gilbert scan, (compressed to 1% width/height)')
#     plt.show()
        
    #alternative, grab the diagonals of the data, might help with memory issue

#%% module check
if __name__ == '__main__':
    
    os.chdir('D:/Documents/__Projects/Image_testing/')
    
    test_image = Image_read("helix_nebula_test.jpg")
    test_image_hls = test_image.image_hls
    test_image_rgb = test_image.image_rgb
    test_image_grey = test_image.image_grey

    
    test_image.threshold(color_space = 'greyscale')
    test_image.gilbert_scan(threshold=True)
    
    test_image_gilbert_grey = test_image.image_gilbert1d_grey_local_threshold
    test_image_grey_global_threshold = test_image.image_grey_local_threshold
    plt.imshow(test_image_grey_global_threshold, cmap='Greys')
    plt.show()
    
    pixel_frequency = np.arange(0, len(test_image_gilbert_grey), 1)
    plt.scatter(pixel_frequency, test_image_gilbert_grey, s=0.2)
    plt.show()
        
        
        
    