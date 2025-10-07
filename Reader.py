# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:28:46 2025

@author: deand
"""

import os


import cv2 as cv          # adaptive thresholding -> in theory, cv can handle everything that skimage uses, well prolly not the save image procedure
import numpy as np        # array handlings
import skimage.color      # global thresholding
import skimage.filters
import skimage.io
import skimage.transform

import matplotlib.pyplot as plt

#import 1d gilbert
import sys
sys.path.insert(1, r"C:\Users\deand\__Github_clones")

import gilbert.gilbert2d as gilbert


class Image_read(object): #read the image in RGB, HSV, and greyscale color space, perhaps switch HSL to HSV
    def __init__(self, filename, message = False):
        if message:
            print('Reading', filename)

        image_array = skimage.io.imread(filename)  #default
        
        #height_width
        self.height = image_array.shape[0]
        self.width = image_array.shape[1]
        self.aspect_ratio = (self.width/self.height)

        #conversion to HLS color space
        hls = cv.cvtColor(image_array, cv.COLOR_RGB2HLS) #check the coordinate again tho
        
        #conversion to uint8
        # gray *=255
        
        self.image_rgb = image_array.astype('uint8')    #image array in rgb color space
        self.image_hls = hls.astype('uint8')            #image array in hls color space
        self.image_grey = hls[:, :, 1].astype('uint8')  #image greyscale, represented by L value


        #preserve for restore -> needs to be copied!
        self.width_old = self.width
        self.height_old = self.height
        self.aspect_ratio_old = self.aspect_ratio
               
        self.image_rgb_old = self.image_rgb.copy()
        self.image_hls_old = self.image_hls.copy()
        self.image_grey_old = self.image_grey.copy()


    def resize (self, width=None, height=None, aspect_preserve=False, message=False):
        if (width is None and height is None):
            return "Height or Width must be specified"
        
        #calculating 
        if aspect_preserve:
            if width is None:
                width = round(height * self.aspect_ratio)
            elif height is None:
                height = round(width * self.aspect_ratio)
            elif (not(np.isclose((width/height), self.aspect_ratio, atol=1e-2))):
                return "Provided dimensions conflict with original aspect ratio"
 
        #use original value for missing dimension
        height = height or self.height
        width = width or self.width
        
        aspect_ratio = width/height
        
        #resize the image -> perform resize on RGB, then convert for HSL color space
        rgb_resize = skimage.transform.resize(self.image_rgb, (height, width))
        rgb_resize *= 255         #convert to uint8
        rgb_resize = rgb_resize.astype('uint8')
        
        hls_resize = cv.cvtColor(rgb_resize, cv.COLOR_RGB2HLS) #conversion to HLS 
        
        #redefine parameters
        if message:
            print("Performing resize to width =", width, "px, height =", height,"px")
        
        self.width = width
        self.height = height
        self.aspect_ratio = aspect_ratio
        
        self.image_rgb = rgb_resize.astype('uint8')
        self.image_hls = hls_resize.astype('uint8')
        self.image_grey = hls_resize[:, :, 1].astype ('uint8')
        
    
    #generate the global and local threshold mask
    def threshold_mask(self, global_threshold_params=np.array([1]),  #global params -> Gaussian blur deviation
                  local_threshold_params=np.array([1,3,2]), #local threshold params -> median blur power, blocksize, constant, see documentations
                  message = False): 
  
        if message:
            print("grabbing boolean local threshold mask")
        #local thresholding process
        median_blur = cv.medianBlur(self.image_grey, local_threshold_params[0])
        local_threshold = cv.adaptiveThreshold(median_blur,1,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,
                                               local_threshold_params[1],
                                               local_threshold_params[2])
 
        if message:
            print("grabbing boolean global threshold mask")
        #global thresholding process
        gaussian_blur = skimage.filters.gaussian(self.image_grey, sigma=global_threshold_params[0])
        threshold = skimage.filters.threshold_otsu(gaussian_blur) #otsu's method seems to be very reliable.
        global_threshold = gaussian_blur < threshold
        
        return global_threshold.astype('bool'), local_threshold.astype('bool')
        
        
    def invert_threshold_mask(threshold_mask, message=False): #invert the threshold mask, keyword = Adaptive
    
        if message:
            print("Inverting local threshold mask")
        return ~threshold_mask
            
            
    def threshold(self, local_mask = None, global_mask = None, color_space=None, message=False): # grab the thresholding
    
        #checking threshold mask    
        if (local_mask is None) and (global_mask is None):
            if message: print('Generating threshold mask')
            global_mask, local_mask = self.threshold_mask()
            
        elif (local_mask is None):
            if message: print('Generating local threshold mask')
            _, local_mask = self.threshold_mask()
            
        elif (global_mask is None):
            if message: print('Generating global threshold mask')
            global_mask, _ = self.threshold_mask()

    
        if color_space == "greyscale" :
            if message:
                print("Performing image threshold in Greyscale color space")
            
            self.image_grey_local_threshold = self.image_grey.copy()
            self.image_grey_local_threshold[local_mask] = 0
            
            self.image_grey_global_threshold = self.image_grey.copy()
            self.image_grey_global_threshold[global_mask] = 0
            
            
            
        elif color_space == "RGB" or color_space == "rgb":
            if message:
                print("Performing image threshold in RGB color space")
                
            self.image_rgb_local_threshold = self.image_rgb.copy()
            self.image_rgb_local_threshold[local_mask,:] = [0,0,0]
            
            self.image_rgb_global_threshold = self.image_rgb.copy()
            self.image_rgb_global_threshold[global_mask,:] = [0,0,0]



        elif color_space == "HLS" or color_space == "hls":
            if message:
                print("Performing image threshold in HLS color space")
                
            self.image_hls_local_threshold = self.image_hls.copy()
            self.image_hls_local_threshold[local_mask,:] = [0,0,0]
            
            self.image_hls_global_threshold = self.image_hls.copy()
            self.image_hls_global_threshold[global_mask,:] = [0,0,0]            
            
            
        else:
            if message:
                print("performing image threshold in Grayscale, HLS, and RGB color space")
                
            self.image_rgb_local_threshold = self.image_rgb.copy()
            self.image_rgb_local_threshold[local_mask,:] = [0,0,0]
            
            self.image_rgb_global_threshold = self.image_rgb.copy()
            self.image_rgb_global_threshold[global_mask,:] = [0,0,0]
            
            self.image_hls_local_threshold = self.image_hls.copy()
            self.image_hls_local_threshold[local_mask,:] = [0,0,0]
            
            self.image_hls_global_threshold = self.image_hls.copy()
            self.image_hls_global_threshold[global_mask,:] = [0,0,0]  
            
            self.image_grey_local_threshold = self.image_grey.copy()
            self.image_grey_local_threshold[local_mask] = 0
            
            self.image_grey_global_threshold = self.image_grey.copy()
            self.image_grey_global_threshold[global_mask]  = 0   
    
        
    def restore (self, size_color=True, message=False):# threshold=True, message=False): #restore the size color and/or thresholding
        if size_color:
            if message:
                print("Restoring original size")
                
            self.width = self.width_old
            self.height = self.height_old
            self.aspect_ratio = self.aspect_ratio_old

            self.image_rgb = self.image_rgb_old
            self.image_hls = self.image_hls_old
            self.image_grey = self.image_grey_old
            # self.image_grayscale = self.image_grayscale_old
        
        # if threshold:             #should be deprecated
        #     try:
        #         self.adapt_threshold = self.adapt_threshold_old
        #         if message:
        #             print("Adaptive threshold restored")
                
        #         self.global_threshold = self.global_threshold_old
        #         if message:
        #             print("Global threshold restored")
                
        #     except (AttributeError):
        #         pass
   
    
    def gilbert_scan (self, color_space = None, message = False, threshold = False): #convert image to 1d using gilbert scan (see Github_link)
        #todo: check with thresholded image
        # scan works, with thresholded image too
            
        x_width = self.width
        y_height = self.height
        
        gilbert_x = []
        gilbert_y = []

        for x, y in gilbert.gilbert2d(x_width, y_height):
            gilbert_x.append(x)
            gilbert_y.append(y)
        
        dim_1d = len(gilbert_x)
        
        
        if color_space == "greyscale" :
            if message:
                print("Performing gilbert scan in Greyscale color space")
            
            self.image_gilbert1d_grey = np.zeros((dim_1d))
            
            for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                self.image_gilbert1d_grey[x] = self.image_grey[gilbert_y[x], gilbert_x[x]]
            
            #test threshold, need failsafe for no thresholded image
            if threshold:
                try:
                    self.image_gilbert1d_grey_local_threshold = np.zeros((dim_1d))
                    self.image_gilbert1d_grey_global_threshold = np.zeros((dim_1d))
                
                    for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                        self.image_gilbert1d_grey_local_threshold[x] = self.image_grey_local_threshold[gilbert_y[x], gilbert_x[x]]
                        self.image_gilbert1d_grey_global_threshold[x] = self.image_grey_global_threshold[gilbert_y[x], gilbert_x[x]]
                
                except AttributeError:
                    print ("No thresholded image in Greyscale color space")
                    pass
            #end test threshold
            
            
        elif color_space == "RGB" or color_space == "rgb":
            if message:
                print("Performing gilbert scan in RGB color space")
            
            self.image_gilbert1d_rgb = np.zeros((dim_1d, 3))
            
            for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                
                self.image_gilbert1d_rgb[x, 0] = self.image_rgb[gilbert_y[x], gilbert_x[x], 0]
                self.image_gilbert1d_rgb[x, 1] = self.image_rgb[gilbert_y[x], gilbert_x[x], 1]
                self.image_gilbert1d_rgb[x, 2] = self.image_rgb[gilbert_y[x], gilbert_x[x], 2]
            
            if threshold:
                 try:
                     self.image_gilbert1d_rgb_global_threshold = np.zeros((dim_1d, 3))
                     self.image_gilbert1d_rgb_local_threshold = np.zeros((dim_1d, 3))

                 
                     for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                         self.image_gilbert1d_rgb_global_threshold[x, 0] = self.image_rgb_global_threshold[gilbert_y[x], gilbert_x[x], 0]
                         self.image_gilbert1d_rgb_global_threshold[x, 1] = self.image_rgb_global_threshold[gilbert_y[x], gilbert_x[x], 1]
                         self.image_gilbert1d_rgb_global_threshold[x, 2] = self.image_rgb_global_threshold[gilbert_y[x], gilbert_x[x], 2]                 
                         
                         self.image_gilbert1d_rgb_local_threshold[x, 0] = self.image_rgb_local_threshold[gilbert_y[x], gilbert_x[x], 0]
                         self.image_gilbert1d_rgb_local_threshold[x, 1] = self.image_rgb_local_threshold[gilbert_y[x], gilbert_x[x], 1]
                         self.image_gilbert1d_rgb_local_threshold[x, 2] = self.image_rgb_local_threshold[gilbert_y[x], gilbert_x[x], 2]                 
                 
                 except AttributeError:
                     print ("No thresholded image in RGB color space")
                     pass       
        
        elif color_space == "HLS" or color_space == "hls":
            if message:
                print("Performing gilbert scan in HLS color space")
            
            self.image_gilbert1d_hls = np.zeros((dim_1d, 3))
            
            for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                
                self.image_gilbert1d_hls[x, 0] = self.image_hls[gilbert_y[x], gilbert_x[x], 0]
                self.image_gilbert1d_hls[x, 1] = self.image_hls[gilbert_y[x], gilbert_x[x], 1]
                self.image_gilbert1d_hls[x, 2] = self.image_hls[gilbert_y[x], gilbert_x[x], 2]          

            if threshold:
                 try:
                     self.image_gilbert1d_hls_global_threshold = np.zeros((dim_1d, 3))
                     self.image_gilbert1d_hls_local_threshold = np.zeros((dim_1d, 3))

                 
                     for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                         self.image_gilbert1d_hls_global_threshold[x, 0] = self.image_hls_global_threshold[gilbert_y[x], gilbert_x[x], 0]
                         self.image_gilbert1d_hls_global_threshold[x, 1] = self.image_hls_global_threshold[gilbert_y[x], gilbert_x[x], 1]
                         self.image_gilbert1d_hls_global_threshold[x, 2] = self.image_hls_global_threshold[gilbert_y[x], gilbert_x[x], 2]                 
                         
                         self.image_gilbert1d_hls_local_threshold[x, 0] = self.image_hls_local_threshold[gilbert_y[x], gilbert_x[x], 0]
                         self.image_gilbert1d_hls_local_threshold[x, 1] = self.image_hls_local_threshold[gilbert_y[x], gilbert_x[x], 1]
                         self.image_gilbert1d_hls_local_threshold[x, 2] = self.image_hls_local_threshold[gilbert_y[x], gilbert_x[x], 2]                 
                 
                 except AttributeError:
                     print ("No thresholded image in HLS color space")
                     pass            
            
        else:
            if message:
                print("performing image threshold in Grayscale, HLS, and RGB color space")
                
            self.image_gilbert1d_grey = np.zeros((dim_1d))
            self.image_gilbert1d_hls = np.zeros((dim_1d, 3))
            self.image_gilbert1d_rgb = np.zeros((dim_1d, 3))
            
            for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                self.image_gilbert1d_grey[x] = self.image_grey[gilbert_y[x], gilbert_x[x]]
                
                self.image_gilbert1d_hls[x, 0] = self.image_hls[gilbert_y[x], gilbert_x[x], 0]
                self.image_gilbert1d_hls[x, 1] = self.image_hls[gilbert_y[x], gilbert_x[x], 1]
                self.image_gilbert1d_hls[x, 2] = self.image_hls[gilbert_y[x], gilbert_x[x], 2]

                self.image_gilbert1d_rgb[x, 0] = self.image_rgb[gilbert_y[x], gilbert_x[x], 0]
                self.image_gilbert1d_rgb[x, 1] = self.image_rgb[gilbert_y[x], gilbert_x[x], 1]
                self.image_gilbert1d_rgb[x, 2] = self.image_rgb[gilbert_y[x], gilbert_x[x], 2]

            if threshold:
                try:
                    self.image_gilbert1d_grey_local_threshold = np.zeros((dim_1d))
                    self.image_gilbert1d_grey_global_threshold = np.zeros((dim_1d))
                
                    for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                        self.image_gilbert1d_grey_local_threshold[x] = self.image_grey_local_threshold[gilbert_y[x], gilbert_x[x]]
                        self.image_gilbert1d_grey_global_threshold[x] = self.image_grey_global_threshold[gilbert_y[x], gilbert_x[x]]
                
                except AttributeError:
                    print ("No thresholded image in Greyscale color space")
                    pass
                
                try:
                     self.image_gilbert1d_rgb_global_threshold = np.zeros((dim_1d, 3))
                     self.image_gilbert1d_rgb_local_threshold = np.zeros((dim_1d, 3))

                 
                     for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                         self.image_gilbert1d_rgb_global_threshold[x, 0] = self.image_rgb_global_threshold[gilbert_y[x], gilbert_x[x], 0]
                         self.image_gilbert1d_rgb_global_threshold[x, 1] = self.image_rgb_global_threshold[gilbert_y[x], gilbert_x[x], 1]
                         self.image_gilbert1d_rgb_global_threshold[x, 2] = self.image_rgb_global_threshold[gilbert_y[x], gilbert_x[x], 2]                 
                         
                         self.image_gilbert1d_rgb_local_threshold[x, 0] = self.image_rgb_local_threshold[gilbert_y[x], gilbert_x[x], 0]
                         self.image_gilbert1d_rgb_local_threshold[x, 1] = self.image_rgb_local_threshold[gilbert_y[x], gilbert_x[x], 1]
                         self.image_gilbert1d_rgb_local_threshold[x, 2] = self.image_rgb_local_threshold[gilbert_y[x], gilbert_x[x], 2]                 
                 
                except AttributeError:
                     print ("No thresholded image in RGB color space")
                     pass  
                 
                try:
                     self.image_gilbert1d_hls_global_threshold = np.zeros((dim_1d, 3))
                     self.image_gilbert1d_hls_local_threshold = np.zeros((dim_1d, 3))

                 
                     for x in range(dim_1d): #gilbert_x and gilbert_y should have the same dimension
                         self.image_gilbert1d_hls_global_threshold[x, 0] = self.image_hls_global_threshold[gilbert_y[x], gilbert_x[x], 0]
                         self.image_gilbert1d_hls_global_threshold[x, 1] = self.image_hls_global_threshold[gilbert_y[x], gilbert_x[x], 1]
                         self.image_gilbert1d_hls_global_threshold[x, 2] = self.image_hls_global_threshold[gilbert_y[x], gilbert_x[x], 2]                 
                         
                         self.image_gilbert1d_hls_local_threshold[x, 0] = self.image_hls_local_threshold[gilbert_y[x], gilbert_x[x], 0]
                         self.image_gilbert1d_hls_local_threshold[x, 1] = self.image_hls_local_threshold[gilbert_y[x], gilbert_x[x], 1]
                         self.image_gilbert1d_hls_local_threshold[x, 2] = self.image_hls_local_threshold[gilbert_y[x], gilbert_x[x], 2]                 
                 
                except AttributeError:
                     print ("No thresholded image in HLS color space")
                     pass            



