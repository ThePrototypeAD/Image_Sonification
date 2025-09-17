# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:02:06 2025

@author: deand
"""

def getval(osc, count=44100, it=False):
    if it: osc = iter(osc)
    # returns 1 sec of samples of given osc.
    return [next(osc) for i in range(count)]