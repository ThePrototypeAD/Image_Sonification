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

#%% sonifying check

if __name__ == '__main__':
    # test_image = Image_read("helix_nebula_test.jpg")
    test_image = Image_read("Saturn.png")
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
    total_wave_unnorm = np.zeros(int(sample_rate*time*len(hue_threshold_local[0])))
    total_wave_norm = np.zeros(int(sample_rate*time*len(hue_threshold_local[0])))
    # duty_check = np.zeros(0)
    
    for h in tqdm(range(len(hue_threshold_local))):
    # for h in range():
        
        init_phase = 0     
        generator_arr_unnorm = np.zeros(0)
        generator_arr_norm = np.zeros(0)
        # duty_check = np.zeros(0)
        for w in (range(len(hue_threshold_local[h]))):
            
            if hue_threshold_local[h, w]<=150:
                duty = 0.5-(hue_threshold_local[h, w]/(2*150))
            elif hue_threshold_local[h, w]>150:
                duty = ((hue_threshold_local[h, w]-150)/(2*30))
            
            #unnormalized
            sq_gen_unnorm = Oscillator(waveform='square', freq=freq_list[h], rate=44100, duty=duty, phase=init_phase)
            gen_points_unnorm = np.array(generate_sample(sq_gen_unnorm, time=time)) 
            gen_points_unnorm = (gen_points_unnorm)*lightness_threshold_local[h, w]
            # init_phase = sq_gen_unnorm.phase # should be the same with normalized and non-normalized
            
            generator_arr_unnorm = np.concatenate((generator_arr_unnorm, gen_points_unnorm), axis=None)
            
            #normalized
            sq_gen_norm = Oscillator(waveform='square_norm', freq=freq_list[h], rate=44100, duty=duty, phase=init_phase)
            gen_points_norm = np.array(generate_sample(sq_gen_norm, time=time)) 
            gen_points_norm = (gen_points_norm)*lightness_threshold_local[h, w]
            
            init_phase = sq_gen_unnorm.phase # should be the same with normalized and non-normalized
            
            generator_arr_norm = np.concatenate((generator_arr_norm, gen_points_norm), axis=None)
        
        total_wave_unnorm = total_wave_unnorm+generator_arr_unnorm
        total_wave_norm = total_wave_norm+generator_arr_norm
    
    total_wave_unnorm = total_wave_unnorm/np.max(abs(total_wave_unnorm))
    total_wave_norm = total_wave_norm/np.max(abs(total_wave_norm))
    
    plt.plot(total_wave_unnorm)
    plt.title('unnormalized')
    plt.show()
    
    plt.plot(total_wave_norm)
    plt.title('normalized')
    plt.show()
    
    to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))
    def wave_to_file(wav, wav2=None, fname="temp", amp=0.1):
        wav = np.array(wav)
        wav = to_16(wav, amp)
        if wav2 is not None:
            wav2 = np.array(wav2)
            wav2 = to_16(wav2, amp)
            wav = np.stack([wav, wav2]).T
    
        wavfile.write(f"D:\Documents\__Projects\sound_test\{fname}.wav", 44100, wav)

    wave_to_file(total_wave_unnorm, fname='image_test_saturn_unnormalized')
    wave_to_file(total_wave_norm, fname='image_test_saturn_normalized')
        
    
    
#%%
#duty check

# if __name__ == "__main__":
#     sample_rate = 44100
#     duty_val = np.linspace(0, 0.5, 20) 
#         #wide duty cycle (>0.5) might be needed as well...
#         # consider the sound, tho...
#         # with the normalized square wave, unsure...
        
#         #again, test the sound!!!
#     time = 0.1
#     total_wave = np.zeros(int(sample_rate*time))
    
#     for vals in duty_val:
#         square_1 = Oscillator(waveform='square_norm', freq=100, rate=44100, duty=vals, phase=0)
#         wave_points = np.array(generate_sample(square_1, time=time)) 
        
#         plt.plot(wave_points, alpha=0.3)
#         total_wave += wave_points
    
#     total_wave = total_wave / np.max(abs(total_wave))
#     plt.plot(total_wave)
#     plt.show()

#%% sound_check
# if __name__ == "__main__":
#     from scipy.io import wavfile
#     sample_rate = 44100
#     duty_val = np.linspace(0, 0.5, 100) #0.01 duty cycle sweep
#     time = 0.1
#     generator_arr = np.zeros(0)
#     total_wave = np.zeros(int(sample_rate*time))
    
#     for vals in duty_val:
#         sq_gen = Oscillator(waveform='square', freq=440, rate=44100, duty=vals, phase=0)#start
#         gen_points = np.array(generate_sample(sq_gen, time=time)) 
            
#         generator_arr = np.concatenate((generator_arr, gen_points), axis=None)
#         # total_wave += gen_points
    
#     # total_wave = total_wave/np.max(abs(total_wave))

#     to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))
#     def wave_to_file(wav, wav2=None, fname="temp", amp=0.1):
#         wav = np.array(wav)
#         wav = to_16(wav, amp)
#         if wav2 is not None:
#             wav2 = np.array(wav2)
#             wav2 = to_16(wav2, amp)
#             wav = np.stack([wav, wav2]).T
    
#         wavfile.write(f"D:\Documents\__Projects\sound_test\{fname}.wav", 44100, wav)

#     wave_to_file(generator_arr, fname='wave_check_duty_cycle_05')
#     # wave_to_file(total_wave, fname='wave_check_duty_cycle_addition_0.5')
    
    
#%% draw array
if __name__ == "__main__":
    import cv2 as cv          # adaptive thresholding -> in theory, cv can handle everything that skimage uses, well prolly not the save image procedure
    import numpy as np        # array handlings
    import skimage.color      # global thresholding
    import skimage.filters
    import skimage.io
    import skimage.transform
    from tqdm import tqdm

    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    
    image_hls = np.zeros((5, 180, 3))
    test_hue = np.linspace(0, 180, 180)
    for i in range(len(image_hls)):
        image_hls[i, :, 0] = test_hue[:]
        image_hls[i, :, 1] = 128
        image_hls[i, :, 2] = 255
    image_hls = image_hls.astype('uint8')
    
    image_rgb = cv.cvtColor(image_hls, cv.COLOR_HLS2RGB) #check the coordinate again tho
    plt.imshow(image_rgb)
    plt.show()
    
    
    #generate above image into simple wave 
    tuning_ratio = [1, 1.25, 1.5, 1.875] # major 7th chord
    test_frequency = 440 * np.array(tuning_ratio)
    
    
    sample_rate = 44100
    # duty_val = np.linspace(0, 0.5, 100) #0.01 duty cycle sweep
    time = 0.25 #0.1 sec per hue value
    total_wave_norm = np.zeros(int(sample_rate*time*len(test_hue)))
    total_wave_unnorm = np.zeros(int(sample_rate*time*len(test_hue)))
    
    hue_to_duty = np.zeros_like(test_hue)
    hue_to_duty[:] = test_hue[:]/360 #from 0 to 0.5
    
    for freq in tqdm(test_frequency):
        init_phase = 0
        generator_arr_unnorm = np.zeros(0)
        generator_arr_norm = np.zeros(0)
        
        for duty in hue_to_duty:
            sq_gen_unnorm = Oscillator(waveform='square', freq=freq, rate=sample_rate, duty=duty, phase=init_phase) #unnormalized
            gen_points_unnorm = np.array(generate_sample(sq_gen_unnorm, time=time)) 
            generator_arr_unnorm = np.concatenate((generator_arr_unnorm, gen_points_unnorm), axis=None)
            
            sq_gen_norm = Oscillator(waveform='square_norm', freq=freq, rate=sample_rate, duty=duty, phase=init_phase) #normalized
            gen_points_norm = np.array(generate_sample(sq_gen_norm, time=time)) 
            generator_arr_norm = np.concatenate((generator_arr_norm, gen_points_norm), axis=None)

        
        total_wave_unnorm += generator_arr_unnorm
        total_wave_norm += generator_arr_norm
    
    #normalized to abs(max)
    total_wave_unnorm = total_wave_unnorm/np.max(abs(total_wave_unnorm))
    total_wave_norm = total_wave_norm/np.max(abs(total_wave_norm))


    #check wave plot
    plt.plot(total_wave_unnorm)
    plt.title('unnormalized square wave test')
    plt.show()
    
    plt.plot(total_wave_norm)
    plt.title('normalized square wave test')
    plt.show()


    to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))
    def wave_to_file(wav, wav2=None, fname="temp", amp=0.1):
        wav = np.array(wav)
        wav = to_16(wav, amp)
        if wav2 is not None:
            wav2 = np.array(wav2)
            wav2 = to_16(wav2, amp)
            wav = np.stack([wav, wav2]).T
    
        wavfile.write(f"D:\Documents\__Projects\sound_test\{fname}.wav", 44100, wav)

    wave_to_file(total_wave_unnorm, fname='squarewave_test_unnormalized')
    wave_to_file(total_wave_norm, fname='squarewave_test_normalized')
    # wave_to_file(total_wave, fname='wave_check_duty_cycle_addition_0.5')
    
    
    