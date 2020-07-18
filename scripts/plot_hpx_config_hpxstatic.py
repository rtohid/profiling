#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:14:34 2018

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
import math
from matplotlib.backends.backend_pdf import PdfPages

now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")


#from matplotlib.backends.backend_pdf import PdfPages

    

directory='/home/shahrzad/Spyder/Blazemark/results/hpx_test/10-13-18-2217'

data_files=glob.glob(directory+'/*.txt')


d={}   

thr=[]
vec_sizes=[]

for filename in data_files:
    th =int(filename.split('_')[-1].split('.')[0])
    vec_size =int(filename.split('_')[-2])
    if th not in thr:
        thr.append(th)
    if vec_size not in vec_sizes:
        vec_sizes.append(vec_size)

thr.sort()
vec_sizes.sort()
    

for v in vec_sizes:
    d[v]=[0]*len(thr)

for filename in data_files:
    f=open(filename, 'r')
    results=f.readlines()

    th =int(filename.split('_')[-1].split('.')[0])
    vec_size =int(filename.split('_')[-2])
    
    time_result=float((results[0].split('time(ns): ')[1]).strip())
    d[vec_size][th-1]=time_result
    
pp = PdfPages(directory+'/hpx_config_performance.pdf')

i=1
for v in vec_sizes:
    plt.figure(i)
    plt.plot(thr, d[v],label='vector size:'+str(v))
    plt.xlabel("# threads")
    plt.ylabel('execution time(ns)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1 
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    
plt.show()
pp.close()