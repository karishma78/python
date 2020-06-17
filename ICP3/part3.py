# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:08:58 2020

@author: chand
"""

import numpy as np
a=np.linspace(1,20,20,dtype=float)
print(a)
b=a.reshape((4,5))
print(b)
b[np.where(b==np.max(b,axis=1,keepdims=True))] = 0.0
print(b);