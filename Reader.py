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



#%%

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
                print("performing image threshold in Grayscale, HSL, and RGB color space")
                
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

  
#%% testing class object
if __name__ == '__main__':
    test_image = Image_read("helix_nebula_test.jpg")
    
    plt.imshow(test_image.image_rgb)
    plt.show()
    plt.imshow(test_image.image_grey, cmap='Greys_r')
    plt.show()
    plt.imshow(test_image.image_hls)
    plt.show()
    
    
    test_image.resize(height=88, aspect_preserve=True)
    plt.imshow(test_image.image_rgb)
    plt.show()
    plt.imshow(test_image.image_grey, cmap='Greys_r')
    plt.show()
    plt.imshow(test_image.image_hls)
    plt.show()
        
    test_mask = np.zeros_like(test_image.image_rgb)
    test_mask = test_mask[:,:,1].astype('bool')
    
    test_image.threshold(local_mask = test_mask, global_mask = test_mask, message=True)
    
    plt.imshow(test_image.image_rgb_global_threshold)
    plt.show()
    plt.imshow(test_image.image_rgb_local_threshold)
    plt.show()    

# global_threshold, local_threshold = test_image.threshold_mask()
# plt.imshow(global_threshold, cmap='Greys_r')
# plt.show()
# plt.imshow(local_threshold, cmap='Greys_r')
# plt.show()  





# test_image.resize(height=19)
# test_image.restore()
# plt.imshow(test_image.image_rgb)
# plt.show()
# plt.imshow(test_image.image_grey, cmap='Greys_r')
# plt.show()
# plt.imshow(test_image.image_hls)
# plt.show()
  
# test_image.resize(height=19)
# test_image.resize(height=302)
# test_image.restore()
# plt.imshow(test_image.image_rgb)
# plt.show()
# plt.imshow(test_image.image_grey, cmap='Greys_r')
# plt.show()
# plt.imshow(test_image.image_hls)
# plt.show()

    
#%% check fx 2
if __name__ == '__main__':
    image = skimage.io.imread("helix_nebula_test.jpg") #check with cv instead of skimage
    
    
    plt.imshow(image)
    plt.show()
    
    image = skimage.transform.resize(image, (88,88))
    image *=255
    image = image.astype('uint8')
    
    plt.imshow(image)
    plt.show()
    
    #convert to HSL
    image_hls = cv.cvtColor(image, cv.COLOR_RGB2HLS)
    image_lightness = image_hls[:, :, 1] #grab lightness value for thresholding
    plt.imshow(image_lightness, cmap='Greys_r')
    plt.title('HLS lightness channel')
    plt.show()

    #check comparison with HSV and greyscale
    image_grey =  skimage.color.rgb2gray(image[:, :, :3])
    plt.imshow(image_grey, cmap='Greys_r')
    plt.title('Greyscale image')
    plt.show()
    
    
    image_hsv  = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    image_value = image_hsv[:, :, 2]
    plt.imshow(image_value, cmap='Greys_r')
    plt.title('HSV value channel')
    plt.show()
    
    ''' 
    three methods are not necessarily equal, but I guess the L channel is viable enough...
    need equation to convert greyscale to L or V channel values
    
    p.s. also deprecating HSV channel for sonification
        -> lightness (black/white) is ambiguous in HSV space
        -> Use HSL (or HLS in cv2 context) to more appropriately represents black/white
    '''
    
    #checking value threshold (global)
    threshold_val = skimage.filters.threshold_otsu(image_lightness)
    threshold_mask = image_lightness < threshold_val
    plt.imshow(threshold_mask, cmap='Greys_r')
    plt.show()
    
    image_copy = image_hls.copy()
    image_copy[threshold_mask,:] = [0,0,0]
    
    # #morphological transform
    # kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    # opening = cv.morphologyEx(threshold_, cv.MORPH_OPEN, kernel)
    
    #view results
    plt.imshow(image_copy, cmap='Greys_r')
    plt.show()
    
    
# after careful consideration, no need to do the length
# just perform the thresholding, then perform the morphological transform!
    
    #checking value threshold (local)
    # median_blur = cv.medianBlur(image_lightness, 1)
    adapt_threshold = cv.adaptiveThreshold(image_lightness,1,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,2)
    plt.imshow(adapt_threshold, cmap='Greys_r')
    plt.show()
    
    image_copy = image_hls.copy()
    adapt_threshold = adapt_threshold.astype('bool')
    image_copy[adapt_threshold,:] = [0,0,0]
    
    #view results
    plt.imshow(image_copy, cmap='Greys_r')
    plt.show()
    

#%% rechecking HSL color space
if __name__ == '__main__':
    image = skimage.io.imread("helix_nebula_test.jpg") #check with cv instead of skimage
    image = skimage.transform.resize(image, (88,88))
    image *= 255 #conversion to uint8
    image = image.astype('uint8')
    
    image_hls = cv.cvtColor(image, cv.COLOR_RGB2HLS) #check the coordinate again tho
    


