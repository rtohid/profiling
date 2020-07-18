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

    

directory='/home/shahrzad/Spyder/Blazemark/results/data/static/10-13-18-2224'

data_files=glob.glob(directory+'/*.dat')


  

thr=[]
vec_sizes=[]

for filename in data_files:
    if 'hpx' in filename:
        vec_size =int(filename.split('-')[-1].split('.')[0])
        th =int(filename.split('-')[-3])
        if th not in thr:
            thr.append(th)
        if vec_size not in vec_sizes:
            vec_sizes.append(vec_size)

thr.sort()
vec_sizes.sort()

d={}     
d_openmp={}
for v in vec_sizes:
    d[v]=[0]*len(thr)
    d_openmp[v]=[0]*len(thr)

for filename in data_files:
    f=open(filename, 'r')
    results=f.readlines()
    if 'hpx' in filename:
        vec_size =int(filename.split('-')[-1].split('.')[0])
        th =int(filename.split('-')[-3])
        
        time_result=float(results[3].split(str(vec_size))[1].strip())
        d[vec_size][th-1]=time_result
    else:
        th =int(filename.split('-')[-2])
        for r in results[3:9]:
            vec_size =int(r.lstrip().split(' ')[0])
            time_result=float(r.split(str(vec_size))[1].strip())
            d_openmp[vec_size][th-1]=time_result



    
pp = PdfPages(directory+'/hpx_config_performance.pdf')

i=1
for v in vec_sizes:
    plt.figure(i)
    plt.title('vector size:'+str(v))
    plt.plot(thr, d[v],label='hpx')
    plt.plot(thr, d_openmp[v], label='openmp')
    plt.xlabel("# threads")
    plt.ylabel('MFlops')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1 
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    
plt.show()
pp.close()