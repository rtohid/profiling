#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:08:05 2018

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")
import math
from matplotlib.backends.backend_pdf import PdfPages



thr=np.arange(1,17).tolist()
chunks=[]
blocks=[]

directory='/home/shahrzad/repos/Blazemark/data/dynamic/log'
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots'
dynamic_date_str='11-08-18-1330'


thr=[]    
block_sizes=[]
chunk_sizes=[]
benchmarks=[]

data_files=glob.glob(directory+'/*.dat')
for filename in data_files:
    b=filename.split('-')[0].split('/')[-1]
    th=int(filename.split('-')[1])
    chunk=int(filename.split('-')[3])
    block=int(filename.split('-')[4].split('.')[0])
    if b not in benchmarks:
        benchmarks.append(b)
    if th not in thr:
        thr.append(th)
    if chunk not in chunk_sizes:
        chunks.append(chunk)
    if block not in block_sizes:
        block_sizes.append(block) 

thr.sort()
block_sizes.sort()
chunk_sizes.sort()
benchmarks.sort()

d_all={}
for benchmark in benchmarks:
    d_all[benchmark]={}
    for b in block_sizes:
        d_all[benchmark][b]={}
        for th in thr:
            d_all[benchmark][b][th]={}
    
for filename in data_files:
    size=[]
    mflops=[]
    f=open(filename, 'r')
    result=f.readlines()[3:163]
    benchmark=filename.split('-')[0].split('/')[-1]
    th=int(filename.split('-')[1])
    chunk=int(filename.split('-')[3])
    block=int(filename.split('-')[4].split('.')[0])
    
   
        
    for r in result:
        size.append(int(r.strip().split(' ')[0]))
        mflops.append(float(r.strip().split(' ')[-1]))
    
    d_all[benchmark][block][th]['size']=size
    d_all[benchmark][block][th]['mflops']=mflops


openmp_date_str='11-15-18-0939'
openmp_directory='/home/shahrzad/repos/Blazemark/data/openmp/log/'+openmp_date_str

i=1
for r in range(1,12):

    data_files=glob.glob(openmp_directory+'/'+str(r)+'-*.dat')
    
    d_openmp={}
    for benchmark in benchmarks:
        d_openmp[benchmark]={}
        for th in thr:
            d_openmp[benchmark][th]={}
        
    for filename in data_files:
        size=[]
        mflops=[]
        f=open(filename, 'r')
        result=f.readlines()[3:163]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])         
            
        for r in result:
            size.append(int(r.strip().split(' ')[0]))
            mflops.append(float(r.strip().split(' ')[-1]))
        
        d_openmp[benchmark][th]['size']=size
        d_openmp[benchmark][th]['mflops']=mflops
    
    
pp = PdfPages(perf_directory+'/perf_log_'+date_str+'.pdf')


for benchmark in benchmarks:
    for b in block_sizes:
        for th in thr:
            plt.figure(i)
        #    plt.title('vector size:'+str(v))            
            plt.title(benchmark+' block size:'+str(b))
            plt.plot(d_all[benchmark][b][th]['size'], d_all[benchmark][b][th]['mflops'],label=str(th)+' threads')
            plt.xlabel("# vector size")           
            plt.ylabel('MFlops')
            plt.xscale('log')
            plt.grid(True, 'both')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)   
        i=i+1
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
    for th in thr:
        plt.figure(i)
    #    plt.title('vector size:'+str(v))            
        plt.title(benchmark+' openmp')
        plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label=str(th)+' threads')
        plt.xlabel("# vector size")           
        plt.ylabel('MFlops')
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
   
plt.show()
pp.close()
