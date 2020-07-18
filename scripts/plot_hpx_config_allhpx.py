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

    
thr=np.arange(1,17).tolist()

directory='/home/shahrzad/Spyder/Blazemark/results/hpx_test/'

data_files=glob.glob(directory+'/*.txt')


d={}   

d_chunk={}
sizes=[100, 1000, 10000, 100000, 1000000, 10000000]
for size in sizes:
    chunk_sizes=[]
    for i in range(int(math.log10(size))):
        chunk_sizes+=[pow(10,i)*k for k in [1,2,3,4,5,6,7,8,9]]
    chunk_sizes.append(size)
    d[size]=[0]*len(chunk_sizes)
    for i in range(len(chunk_sizes)):
        d[size][i]=[0]*len(thr)
    d_chunk[size]=chunk_sizes

for filename in data_files:
    th =int(filename.split('_')[-1].split('.')[0])
    chunk_size =int(filename.split('_')[-2])
    vec_size =int(filename.split('_')[-3])    
    
    f=open(filename, 'r')
    results=f.readlines()
    if len(results)>3*(th+1)+1:
        results=results[3*(th+1)+1:]
        
    time_result=float((results[0].split('time(ns): ')[1]).strip())
    d[vec_size][d_chunk[vec_size].index(chunk_size)][thr.index(th)]=time_result
    
pp = PdfPages(directory+'/hpx_config_performance.pdf')

i=1
for size in sizes:
    plt.figure(i)
    for chunk_size in d_chunk[size]:
        plt.plot(thr, d[size][d_chunk[size].index(chunk_size)],label='vector size:'+str(size)+' chunk size:'+str(chunk_size))
    plt.xlabel("# threads")
    plt.ylabel('execution time(ns)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1 
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')

for size in sizes:
    plt.figure(i)
    for th in thr:
        plt.xscale('log')
        plt.plot(d_chunk[size], [c[th-1] for c in d[size]],label='vector size:'+str(size)+' thread:'+str(th))
    plt.xlabel("chunk size")
    plt.ylabel('execution time(ns)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1 
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    
plt.show()
pp.close()