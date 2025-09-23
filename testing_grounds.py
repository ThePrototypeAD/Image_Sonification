# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:11:11 2025

@author: deand
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#importing image reader module
import sys
sys.path.insert(1, 'D:\Documents\__Projects')

from Image_Sonification.Reader import Image_read
from Image_Sonification.Oscillator_simple import Oscillator
from Image_Sonification.Oscillator_simple import frequency_gen
from Image_Sonification.Oscillator_simple import generate_sample



#start sonifying
os.chdir('D:/Documents/__Projects/Image_testing/')

if __name__ == '__main__':
    test_image = Image_read("helix_nebula_test.jpg")
    hls_image = test_image.image_hls
    
    # plt.imshow(hls_image)
    # plt.show()
    
    test_image.resize(height=88, aspect_preserve=True)
    test_image.threshold()
    
    hls_threshold_local = test_image.image_hls_local_threshold
    hls_threshold_global = test_image.image_hls_global_threshold
    
    # plt.imshow(hls_threshold_local)
    # plt.show()
    
    
    hue_threshold_local = hls_threshold_local[:, :, 0]
    lightness_threshold_local = hls_threshold_local[:, :, 1]
    saturation_threshold_local = hls_threshold_local[:, :, 2]
    
    #max-normalize lightness and saturation to 1
    lightness_threshold_local = lightness_threshold_local / np.max(lightness_threshold_local)
    saturation_threshold_local = saturation_threshold_local / np.max(saturation_threshold_local)
    
    
    #test lightness and hue first
    freq_list = frequency_gen()
    # freq_list = freq_list[::-1] #invert list
    
    

    
    sample_rate = 44100
    time = 0.1 #(0.1 second for each ) horizontal pixel (try this with bpm later)
    total_wave = np.zeros(int(sample_rate*time*len(hue_threshold_local[0])))
    # duty_check = np.zeros(0)
    
    for h in tqdm(range(len(hue_threshold_local))):
    # for h in range():
        
        init_phase = 0     
        generator_arr = np.zeros(0)
        # duty_check = np.zeros(0)
        for w in (range(len(hue_threshold_local[h]))):
            
            if hue_threshold_local[h, w]<=150:
                duty = 0.5-(hue_threshold_local[h, w]/(2*150))
            elif hue_threshold_local[h, w]>150:
                duty = ((hue_threshold_local[h, w]-150)/(2*30))
            
            
            sq_gen = Oscillator(waveform='square', freq=freq_list[h], rate=44100, duty=0.5, phase=init_phase)#start
            gen_points = np.array(generate_sample(sq_gen, time=time)) 
            gen_points = (gen_points)*lightness_threshold_local[h, w]
            init_phase = sq_gen.phase
            
            generator_arr = np.concatenate((generator_arr, gen_points), axis=None)
            # duty_check = np.(duty_check, duty)
        
        # plt.plot(generator_arr)
        # plt.show()
        
        total_wave = total_wave+generator_arr
    plt.plot(total_wave)
    plt.show()
        
    
    
#%%
#duty check

if __name__ == "__main__":
    sample_rate = 44100
    duty_val = np.linspace(0, 1, 20) 
        #wide duty cycle (>0.5) might be needed as well...
        # consider the sound, tho...
    time = 0.1
    total_wave = np.zeros(int(sample_rate*time))
    
    for vals in duty_val:
        square_1 = Oscillator(waveform='square', freq=100, rate=44100, duty=vals, phase=0)
        wave_points = np.array(generate_sample(square_1, time=time)) 
        
        plt.plot(wave_points, alpha=0.3)
        total_wave += wave_points
    
    total_wave = total_wave / np.max(abs(total_wave))
    plt.plot(total_wave)
    plt.show()

#%% sound_check
if __name__ == "__main__":
    from scipy.io import wavfile
    sample_rate = 44100
    duty_val = np.linspace(0, 0.5, 100) #0.01 duty cycle sweep
    time = 0.1
    generator_arr = np.zeros(0)
    total_wave = np.zeros(int(sample_rate*time))
    
    for vals in duty_val:
        sq_gen = Oscillator(waveform='square', freq=440, rate=44100, duty=vals, phase=0)#start
        gen_points = np.array(generate_sample(sq_gen, time=time)) 
            
        generator_arr = np.concatenate((generator_arr, gen_points), axis=None)
        # total_wave += gen_points
    
    # total_wave = total_wave/np.max(abs(total_wave))

    to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))
    def wave_to_file(wav, wav2=None, fname="temp", amp=0.1):
        wav = np.array(wav)
        wav = to_16(wav, amp)
        if wav2 is not None:
            wav2 = np.array(wav2)
            wav2 = to_16(wav2, amp)
            wav = np.stack([wav, wav2]).T
    
        wavfile.write(f"D:\Documents\__Projects\sound_test\{fname}.wav", 44100, wav)

    wave_to_file(generator_arr, fname='wave_check_duty_cycle_05')
    # wave_to_file(total_wave, fname='wave_check_duty_cycle_addition_0.5')
    
    
#%% draw array
if __name__ == "__main__":
    import cv2 as cv          # adaptive thresholding -> in theory, cv can handle everything that skimage uses, well prolly not the save image procedure
    import numpy as np        # array handlings
    import skimage.color      # global thresholding
    import skimage.filters
    import skimage.io
    import skimage.transform

    import matplotlib.pyplot as plt
    
    image_hls = np.zeros((10, 180, 3))
    test_hue = np.linspace(0, 180, 180)
    for i in range(10):
        image_hls[i, :, 0] = test_hue[:]
        image_hls[i, :, 1] = 128
        image_hls[i, :, 2] = 255
    image_hls = image_hls.astype('uint8')
    
    image_rgb = cv.cvtColor(image_hls, cv.COLOR_HLS2RGB) #check the coordinate again tho
    plt.imshow(image_rgb)
    plt.show()
    