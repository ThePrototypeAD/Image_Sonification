# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 17:06:19 2025

@author: deand
"""

import sys
sys.path.insert(1, 'D:\Documents\__Projects')

#image reader components
from Image_Sonification.Reader import Image_read

#synthesizer components
from Synthesizer2.components import composers as synth_comp
from Synthesizer2.components import envelopes as synth_envs
from Synthesizer2.components import modifiers as Synth_mods

from Synthesizer2.components.oscillators import oscillators as synth_osc
from Synthesizer2.components.oscillators import modulated_oscillator as synth_osc_mod

import numpy as np


#%% make it a function
def frequency_gen (tuning_freq = 440,      # base tuning frequency
                   tuning_ratio = None,    # use this custom tuning, if kept Nonetype, proceed to equal temperament accepts array!
                   tones = 12,             # how many keys are in an octave
                   keys = 88,              # how many total keys if keys == all or max, do all frequencies.
                   lowest_limit = 20,      # lowest limit of human hearing
                   highest_limit = 20000,  # highest limit of human hearing
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
            tuning_ratio = np.array(tuning_ratio).astype(float)
            print ('using provided tuning ratio = ', tuning_ratio)
            print()
        
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
        return tuning_freq*all_intervals
    
    else:
        all_intervals = all_intervals[0:keys]
        return tuning_freq*all_intervals


#%% class to sonify, asumme the image is the controller.
# 3 inputs, HSV + 1 time - pixel


#%% testing_grounds

if __name__ == '__main__':
    test = [1]
    test1 = np.array([1, 1.2, 1.5])
    tone_gen = frequency_gen(tuning_ratio = 'test', messages = True)







