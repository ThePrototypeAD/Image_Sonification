# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 17:01:57 2025

@author: deand
"""

import numpy as np

class Oscillator:
    def __init__(self, waveform='sine', freq=440, rate=44100, duty=0.5, phase=0.0):
        self.freq = freq
        self.rate = rate
        self.phase = phase % 1.0 
        self.increment = freq / rate
        self.waveform = waveform
        self.duty = duty  # For square wave


    def __call__(self):
        t = self.phase

        # Select waveform
        if self.waveform == 'sine':
            val = np.sin(2 * np.pi * t)

        elif self.waveform == 'saw':
            val = 2 * (t % 1.0) - 1  # range [-1, 1]

        elif self.waveform == 'triangle':
            val = 2 * abs(2 * (t % 1.0) - 1) - 1  # range [-1, 1]

        elif self.waveform == 'square':
            val = 1.0 if (t % 1.0) < self.duty else -1.0


        else:
            raise ValueError(f"Unsupported waveform: {self.waveform}")

        # Advance phase
        self.phase += self.increment
        if self.phase >= 1.0:
            self.phase -= 1.0

        return val

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    def getval_np2(osc, count=44100): #generate 1 second sample by default
        # Check if osc is callable (function), else assume iterator
        if callable(osc):
            return [osc() for _ in range(count)]
        else:
            # osc is iterator or iterable
            if not hasattr(osc, '__next__'):
                osc = iter(osc)
            return [next(osc) for _ in range(count)]
    
    points = int(44100/5)
    
    test_square1 = Oscillator(waveform='square', freq=12, rate=44100, duty=0.5, phase=0.0)#start
    s1 = getval_np2(test_square1, count=points)[:points]
    
    test_square2 = Oscillator(waveform='square', freq=12, rate=44100, duty=0.5, phase=test_square1.phase)#start
    s2 = getval_np2(test_square2, count=points)[:points]
    
    s_all = np.concatenate((s1, s2), axis=None)
    plt.plot(s_all)
    plt.show()


from tqdm import tqdm 

#modifying getval function
def getval_np1(osc, count=44100):
    if not hasattr(osc, '__next__'):
        osc = iter(osc)
    return np.fromiter((next(osc) for _ in range(count)), dtype=np.float32)

def getval_np(osc, count=44100, it=False):
    if it: osc = iter(osc)
    return np.fromiter((next(osc) for _ in range(count)), dtype=np.float32)

def getval_np2(osc, count=44100):
    # Check if osc is callable (function), else assume iterator
    if callable(osc):
        return [osc() for _ in range(count)]
    else:
        # osc is iterator or iterable
        if not hasattr(osc, '__next__'):
            osc = iter(osc)
        return [next(osc) for _ in range(count)]

#checking benchmark





#initialize the array
wave_point = int(44100/10) #0.1 seconds for each sample
generator_arr = np.zeros(0)

def getval_np2(osc, count=44100): #generate 1 second sample by default
        # Check if osc is callable (function), else assume iterator
        if callable(osc):
            return [osc() for _ in range(count)]
        else:
            # osc is iterator or iterable
            if not hasattr(osc, '__next__'):
                osc = iter(osc)
            return [next(osc) for _ in range(count)]
    
points = int(44100/10)

hue_arr = np.linspace(0, 180, num=360) #using the cv2 hue calculation, 0-180

duty_arr = np.zeros_like(hue_arr)

for i in range(len(duty_arr)):
    if hue_arr[i]<=150: 
        duty_arr[i] = (-hue_arr[i]/(2*150))+0.5
    elif hue_arr[i]>150:
        duty_arr[i] = ((hue_arr[i]-150)/(2*30))




#make the 0.5 = red, 
init_phase = 0

for i in tqdm(range(len(duty_arr))):

    gen = Oscillator(waveform='square', freq=440, rate=44100, duty=duty_arr[i], phase=init_phase)#start
    gen_points = (getval_np2(gen)[:wave_point]) 
    init_phase = gen.phase
    
        # generating points for gen is slow... -> generator itself is kinda slow... time to optimize this to my use
        # square oscillation is slow!

    generator_arr = np.concatenate((generator_arr, gen_points), axis=None)

wave_to_file(generator_arr, fname='check1_chatgpt') #this is correct
# todo : employ low pass filter
    
    
    