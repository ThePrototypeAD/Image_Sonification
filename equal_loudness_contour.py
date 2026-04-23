# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:24:42 2026

@author: deand

testing the equal loudness contour table

adapted from: https://crackedbassoon.com/writing/equal-loudness-contours

introduction 
1. phon = perceived equal loudness
frequencies with differing SPL, but having the same phon are perceived as having equal loudness
using phon might be more beneficial than directly translating intensity to volume as perceived difference might be more important than actual difference

"""
#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rcParams as defaults

#%% importing ISO222 table
# credit 1: https://www.dsprelated.com/showcode/174.php,
# credit 2: https://crackedbassoon.com/writing/equal-loudness-contours

# if only there's some csv file... we can load this with pandas
table_f = np.array(
    [
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
    ]
)
table_af = np.array(
    [
        0.532,
        0.506,
        0.480,
        0.455,
        0.432,
        0.409,
        0.387,
        0.367,
        0.349,
        0.330,
        0.315,
        0.301,
        0.288,
        0.276,
        0.267,
        0.259,
        0.253,
        0.250,
        0.246,
        0.244,
        0.243,
        0.243,
        0.243,
        0.242,
        0.242,
        0.245,
        0.254,
        0.271,
        0.301,
    ]
)
table_Lu = np.array(
    [
        -31.6,
        -27.2,
        -23.0,
        -19.1,
        -15.9,
        -13.0,
        -10.3,
        -8.1,
        -6.2,
        -4.5,
        -3.1,
        -2.0,
        -1.1,
        -0.4,
        0.0,
        0.3,
        0.5,
        0.0,
        -2.7,
        -4.1,
        -1.0,
        1.7,
        2.5,
        1.2,
        -2.1,
        -7.1,
        -11.2,
        -10.7,
        -3.1,
    ]
)
table_Tf = np.array(
    [
        78.5,
        68.7,
        59.5,
        51.1,
        44.0,
        37.5,
        31.5,
        26.5,
        22.1,
        17.9,
        14.4,
        11.4,
        8.6,
        6.2,
        4.4,
        3.0,
        2.2,
        2.4,
        3.5,
        1.7,
        -1.3,
        -4.2,
        -6.0,
        -5.4,
        -1.5,
        6.0,
        12.6,
        13.9,
        12.3,
    ]
)

#%% grab the elc function
# credit: https://crackedbassoon.com/writing/equal-loudness-contours
#limit -> only 0-90 phon, and 20-12.5k hz limitation

def equal_loudness_contour(phon): # phon = raw loudness from data
    assert 0 <= phon <= 90, f"phon {phon} is not within 0-90"
    Ln = phon
    
    #equation from ISO226
        #piecewise to make power clearer
    powerLn = np.power(10, 0.025*Ln)
    powerTfLu = np.power(10, ((table_Tf + table_Lu)/10)-9)
    Af = 4.47e-3 * (powerLn - 1.15) + np.power(0.4 * powerTfLu, table_af)
    
    Lp = ((10.0 / table_af) * np.log10(Af)) - table_Lu + 94

    return Lp

#%% specific frequency
def equal_loudness_frequency(phon, frequency):
    
    target_lp = equal_loudness_contour(phon)
    
    #lambda fx to check boolean for assertion
    min_assert = lambda freq : freq >= table_f.min()
    max_assert = lambda freq : freq <= table_f.max()

    frequency = np.array(frequency) #in case one forgor to put the datatype to array
        
    assert min_assert(frequency.min()), f"frequency too low (under {table_f.min()} Hz)"
    assert max_assert(frequency.max()), f"frequency too high (over {table_f.max()} Hz)"
   
    tck = interpolate.splrep(table_f, target_lp, s=0)
    Lp = interpolate.splev(frequency, tck, der=0)
    
    return Lp

#%% test plot
def plot_elcs():
    """Makes the equal-loudness-contour plot.

    """
    defaults["lines.linewidth"] = 2
    defaults["font.size"] = 14

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    # x = np.logspace(np.log10(table_f.min()), np.log10(table_f.max()), 1000)

    for p in range(0, 100, 10):
        c, l = ("C0", None) if p != 60 else ("C1", "60 phon")
        ax.plot(table_f, equal_loudness_contour(p), c=c, label=l)

    ax.legend(fancybox=False, framealpha=0)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Sound pressure level (dB)")

    
#%% plotting elc if main
if __name__ == "__main__": #testing before importing

    plot_elcs()
    
    frequencies_test = np.array([100, 1000, 1000000000])
    frequency_test = 440
    
    ELC_freq_arr = equal_loudness_frequency(60, frequencies_test)
    ELC_freq_int = equal_loudness_frequency(60, frequency_test)

