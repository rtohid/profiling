#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:42:16 2019

@author: shahrzad
"""
import numpy as np
from matplotlib import pyplot as plt
plt.figure(1)
ax = plt.subplot(111)

sigma=0.2
kappa=.050
e=15
p=np.linspace(1,e,e)
y=p/(1+sigma*(p-1))
z=p/(1+sigma*(p-1)+kappa*p*(p-1))
plt.plot(p,p,label="Linear Speedup")
plt.plot(p,y,label="Amdahl's Law")
plt.plot(p,z,label="USL")
ax.set_xticks(np.arange(2,e,2))
ax.set_xticklabels(np.arange(2,e,2))
ax.set_yticks(np.arange(2,e,2))
ax.set_yticklabels(np.arange(2,e,2))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
plt.savefig('/home/shahrzad/src/Dissertation/images/USL_1.png',dpi=300,bbox_inches='tight')
