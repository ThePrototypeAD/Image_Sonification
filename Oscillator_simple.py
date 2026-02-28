# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 17:01:57 2025

@author: deand
"""

import numpy as np
import scipy.signal as sig
from scipy.io import wavfile

#%%  Oscillator Skeleton
class Oscillator:               #skeleton 
    def __init__(self, waveform='sine', freq=440, rate=44100, duty=0.5, phase=0.0): #phase 0-1, 
        self.freq = freq
        self.rate = rate
        self.phase = phase % 1.0 
        self.increment = freq / rate
        self.waveform = waveform
        self.duty = duty  #square wave duty cycle


    def __call__(self):
        t = self.phase

        # Select waveform
        if self.waveform == 'sine':
            val = np.sin(2 * np.pi * t)

        elif self.waveform == 'saw':
            val = 2 * (t % 1.0) - 1  

        elif self.waveform == 'triangle':
            val = 2 * abs(2 * (t % 1.0) - 1) - 1  

        elif self.waveform == 'square':
            val = 1.0 if (t % 1.0) < self.duty else -1.0
        
        elif self.waveform == 'square_norm': # -> normalized square wave
                                             # need failsafe is self.duty = 0
                            
            val = 1.0 if (t % 1.0) < self.duty else (self.duty/(1-self.duty))*-1.0 

        # else:
        #     raise ValueError(f"Unsupported waveform: {self.waveform}")

        # Advance phase
        self.phase += self.increment
        if self.phase >= 1.0:
            self.phase -= 1.0

        return val



#%% frequency generator
def frequency_gen (tuning_freq = 440,      # base tuning frequency
                   tuning_ratio = None,    # use this custom tuning, if kept Nonetype, proceed to equal temperament accepts array!
                   tones = 12,             # how many keys are in an octave
                   keys = 88,              # how many total keys if keys == all or max, do all frequencies.
                   lowest_limit = 20,      # lowest limit of human hearing
                   highest_limit = 20000,  # highest limit of human hearing
                   invert_freq = True,      # invert the order of frequency
                   messages = False):       # enable/disable print messages

#if left blank, frequency_gen will generate ordinary 12-TET, with tuning frequency A4=440 
# allows microtonality with the semitone_octave variable   
# limitation #1 : lowest possible frequency is always on, upper frequencies are sacrified
# limitation #2 : can only do equal temperament. Any other style of tuning are not yet supported
#        fixed limitation #2

    # grabbing the possible amount of octaves 
    # also check the lowest and highest frequency
    lowest_frequency = tuning_freq
    highest_frequency = tuning_freq
    div_power = 0
    mult_power = 0

    #check how many octaves are needed
    while lowest_frequency > (lowest_limit*2):
        lowest_frequency = lowest_frequency/2
        div_power -=1


    while highest_frequency < (highest_limit/2):
        highest_frequency = highest_frequency*2
        mult_power += 1
    
    octaves = np.arange(div_power, mult_power).astype(float)
    octaves_freq_mult = np.power(2, octaves)
    
    
    #grabing the intervals according to the tones
    
    if isinstance(tuning_ratio, (list, np.ndarray)): #check if tuning ratio is provided in the type of list or nd.array
        if messages:
            print ('using provided tuning ratio = ', tuning_ratio)
            print()
        tuning_ratio = np.array(tuning_ratio).astype(float)
        
        if len(tuning_ratio) < 2 or np.max(tuning_ratio)>=2 or np.min(tuning_ratio)<1:
            return print('Tuning ratio provided but is invalid. Check the tuning ratio')
    
    elif tones > 0 and tuning_ratio == None:
        if messages:
            print('no tuning ratio provided.')
            print('using', tones, "tones equal temperament")
            print()
        
        interval = np.arange(0, tones)
        tuning_ratio = np.power(2, interval / len(interval))
    
    else: 
        return print('invalid equal temperament tones or tuning ratio')
               

    all_intervals = np.concatenate([tuning_ratio * octaves_freq_mult 
                                    for octaves_freq_mult in octaves_freq_mult], 
                                   axis=None)

    if (keys == 'all' or keys == 'max'):
        return tuning_freq*all_intervals

    elif len(all_intervals) < keys:
        if messages:
            print ("Current setting is inaudible to human hearing")
            print ("using possible maximum number of keys = ", len(all_intervals), " keys")
        result_freq = tuning_freq*all_intervals
    
    else:
        all_intervals = all_intervals[0:keys]
        result_freq = tuning_freq*all_intervals
    
    if invert_freq:
        return result_freq[::-1]
    else : 
        return result_freq
    

#%% hue to duty functions

#auxiliary fxs
#inside the cutoff
def duty_res1 (hue_input, cutoff_high, cutoff_low, constant):
    return ((cutoff_high-hue_input)*constant)/(cutoff_high - cutoff_low)
    
#outside the cutoff
def duty_res2 (hue_input, cutoff_high, cutoff_low, constant, max_val=360):
    return (-(cutoff_high-hue_input)*constant)/(max_val-(cutoff_high - cutoff_low))


#define function while outside of hue range
def hue_to_duty (hue_input, 
                   hue_cutoff_low, 
                   hue_cutoff_high,
                   hue_max = 360, 
                   duty_min = 0, 
                   duty_max = 50, 
                   poly_degree_1=1, 
                   poly_degree_2=1, 
                   invert = False):


    #start the trick
    hue_calculate = hue_input.copy()

    try: #default calculation for nonarray input
        #if hue lower than low cutoff, not, wrap the value as if + hue_max
        hue_calculate = hue_calculate + hue_max if hue_calculate < hue_cutoff_low else hue_calculate
    
        if hue_calculate <= hue_cutoff_high:
            constant_poly = np.power((duty_max - duty_min), 1/poly_degree_1)
            duty_val = np.power(duty_res1(hue_calculate, hue_cutoff_high, hue_cutoff_low, constant_poly), poly_degree_1)
    
        else: 
            constant_poly = np.power((duty_max - duty_min), 1/poly_degree_2)
            duty_val = np.power(duty_res2(hue_calculate, hue_cutoff_high, hue_cutoff_low, constant_poly, hue_max), poly_degree_2)


    
    except ValueError:        #if array
        for i in range(len (hue_calculate)):
            hue_calculate[i] = hue_calculate[i] + hue_max if (hue_calculate[i] <hue_cutoff_low) else hue_calculate[i]
    
        constant_poly1 = np.power((duty_max - duty_min), 1/poly_degree_1)
        constant_poly2 = np.power((duty_max - duty_min), 1/poly_degree_2)
    
        duty_val = np.where(hue_calculate <= hue_cutoff_high, \
                            (np.power(duty_res1(hue_calculate, hue_cutoff_high, hue_cutoff_low, constant_poly1), poly_degree_1)),
                            (np.power(duty_res2(hue_calculate, hue_cutoff_high, hue_cutoff_low, constant_poly2, hue_max), poly_degree_2)))

    
    if invert:
        duty_result = -(duty_val+duty_min) + duty_max
    else:
        duty_result = duty_val+duty_min
        
    return duty_result/100 #convert to percent


#%% saturation field


    
#todo -> saturation - not low pass... but check various ideas...
#change the saturation to the companying oscillator (or perhaps put an LFO?)
# depending on how fast the sample, we need lesser LFO
def saturation_frequency (saturation, base_frequency, freq_arr, power = 1):
    #saturation is from 0 to 1, limit to 50 cents
    # freq_arr is guaranteed 1 dimension array
    
    freq_arr_index = np.argwhere(freq_arr == base_frequency)
    freq_arr_index = freq_arr_index[0]
    freq_arr_index = freq_arr_index[0]
    try:
        freq_ratio = base_frequency/freq_arr[freq_arr_index+1]
    except IndexError: #should be fine in an equal temprament, but unequal temprament will introduce some jank
        freq_ratio = freq_arr[freq_arr_index-1]/base_frequency 
    
    function_constant = (freq_ratio-1)*0.5 # -> limit at 50 cents, scaled from next frequency ratio
    # 1+fx constant = 50 cents of interval
    # todo: See how different cents affect different sound perception
    
    
    # anchor form = a(x^n)+a+1 = keeps the 1 saturation to 1, a is function of frequency ratio
    # anchor at (1, 1) -> Saturation of 1 = 1 
    
    sub_ratio = -function_constant*(np.power(saturation, power))+function_constant+1
        #limit to half of ratio -> invert the freq to get upper frequency
        
    sub_frequency = (base_frequency*sub_ratio)
    
    return sub_frequency



#%% sound-generating function

def generate_sample (osc, sample_rate = 44100, time = 1): 
        # generate sample for the oscillator
        #time = in second, how many sample rate in a second
    if callable(osc):
        return [osc() for _ in range(int(sample_rate*time))]
    else:
        # osc is iterator or iterable
        if not hasattr(osc, '__next__'):
            osc = iter(osc)
        return [next(osc) for _ in range(int(sample_rate*time))]

def wave_to_file(wav, wav2=None, fname="temp", amp=0.1):
    to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 is not None: #for stereo
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    wavfile.write(f"D:\Documents\__Projects\sound_test\{fname}.wav", 44100, wav)
        

#%% utils





#%%
if __name__ == '__main__':
    
    test_freq_arr = frequency_gen()
    
    test_freq = test_freq_arr[28]
    
    index = np.argwhere(test_freq_arr == test_freq)
    index = index[0]
    index = index[0]
    
    print(index)
    
    sub_freq = saturation_frequency(0, test_freq, test_freq_arr)
    
    print(sub_freq/44100)
    print(test_freq_arr[28])
    
    # import matplotlib.pyplot as plt
    
    # def getval_np2(osc, count=44100): #generate 1 second sample by default
    #     # Check if osc is callable (function), else assume iterator
    #     if callable(osc):
    #         return [osc() for _ in range(count)]
    #     else:
    #         # osc is iterator or iterable
    #         if not hasattr(osc, '__next__'):
    #             osc = iter(osc)
    #         return [next(osc) for _ in range(count)]
    
    # points = int(44100/5)
    
    # test_square1 = Oscillator(waveform='square', freq=12, rate=44100, duty=0.5, phase=0.0)#start
    # s1 = getval_np2(test_square1, count=points)[:points]
    
    # test_square2 = Oscillator(waveform='square', freq=12, rate=44100, duty=0.5, phase=test_square1.phase)#start
    # s2 = getval_np2(test_square2, count=points)[:points]
    
    # s_all = np.concatenate((s1, s2), axis=None)
    # plt.plot(s_all)
    # plt.show()

    
    # from tqdm import tqdm 
    
    # #modifying getval function
    # def getval_np1(osc, count=44100):
    #     if not hasattr(osc, '__next__'):
    #         osc = iter(osc)
    #     return np.fromiter((next(osc) for _ in range(count)), dtype=np.float32)
    
    # def getval_np(osc, count=44100, it=False):
    #     if it: osc = iter(osc)
    #     return np.fromiter((next(osc) for _ in range(count)), dtype=np.float32)
    
    # def getval_np2(osc, count=44100):
    #     # Check if osc is callable (function), else assume iterator
    #     if callable(osc):
    #         return [osc() for _ in range(count)]
    #     else:
    #         # osc is iterator or iterable
    #         if not hasattr(osc, '__next__'):
    #             osc = iter(osc)
    #         return [next(osc) for _ in range(count)]
    
    # #checking benchmark
    
    
    
    
    
    # #initialize the array
    # wave_point = int(44100/10) #0.1 seconds for each sample
    # generator_arr = np.zeros(0)
    
    # def getval_np2(osc, count=44100): #generate 1 second sample by default
    #         # Check if osc is callable (function), else assume iterator
    #         if callable(osc):
    #             return [osc() for _ in range(count)]
    #         else:
    #             # osc is iterator or iterable
    #             if not hasattr(osc, '__next__'):
    #                 osc = iter(osc)
    #             return [next(osc) for _ in range(count)]
        
    # points = int(44100/10)
    
    # hue_arr = np.linspace(0, 180, num=360) #using the cv2 hue calculation, 0-180
    
    # duty_arr = np.zeros_like(hue_arr)
    
    # for i in range(len(duty_arr)):
    #     if hue_arr[i]<=150: 
    #         duty_arr[i] = (-hue_arr[i]/(2*150))+0.5
    #     elif hue_arr[i]>150:
    #         duty_arr[i] = ((hue_arr[i]-150)/(2*30))
    
    
    
    
    # #make the 0.5 = red, 
    # init_phase = 0
    
    # for i in tqdm(range(len(duty_arr))):
    
    #     gen = Oscillator(waveform='square', freq=440, rate=44100, duty=duty_arr[i], phase=init_phase)#start
    #     gen_points = (getval_np2(gen)[:wave_point]) 
    #     init_phase = gen.phase
        
    #         # generating points for gen is slow... -> generator itself is kinda slow... time to optimize this to my use
    #         # square oscillation is slow!
    
    #     generator_arr = np.concatenate((generator_arr, gen_points), axis=None)
    
    # wave_to_file(generator_arr, fname='check1_chatgpt') #this is correct
    # # todo : employ low pass filter
    
    
    