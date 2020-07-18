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
now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")


from matplotlib.backends.backend_pdf import PdfPages

    
thr=np.arange(16).tolist()
#sizes=[100,1000,10000,100000,1000000,10000000]
sizes=[1000000]
if len(sizes)==1:
    directory='/home/shahrzad/Spyder/Blazemark/results/data/'+str(sizes[0])+'/3'
else:
    directory='/home/shahrzad/Spyder/Blazemark/results/data/'
data_files=sorted(glob.glob(directory+'/*.dat'))

d_all={}
d_openmp={}
for s in sizes:
    d_all[s]={}
    d_openmp[s]={}
    for t in thr:
        d_all[s][t]=[[],[]]
        d_openmp[s][t]=[[1],[]]


multiplyers=[]    
chunk_sizes=[]    

for filename in data_files:
    if filename.count('-')==2:
        (benchmark, cores, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        print(benchmark, cores, runtime)
        k=runtime
    else:
        (benchmark, cores, runtime, chunk_size, multiplyer, threshold) = filename.split('/')[-1].replace('.dat','').split('-')
        print(benchmark, cores, runtime, chunk_size, multiplyer, threshold)
        
        k=chunk_size+'-'+multiplyer+'-'+threshold
        if multiplyer not in multiplyers:
            multiplyers.append(multiplyer)
            
        if chunk_size not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))
            
    f=open(filename, 'r')
    results=f.readlines()[3:9]

    for r in results:
        if len(sizes)==1:
            if '     '+str(sizes[0])+' ' in r:
                s=r.strip().split(' ')[0]
                print("s:",s)
                if 'openmp' in filename:                
                    d_openmp[int(s)][int(cores)-1][1]=float(r.strip().split(' ')[-1])
                else:
                    d_all[int(s)][int(cores)-1][0].append(float(chunk_size))
                    d_all[int(s)][int(cores)-1][1].append(float(r.strip().split(' ')[-1]))

 
for m in multiplyers:
    pp = PdfPages(directory+'/blazemark_hpx_config_performance_threads.pdf')

    i=1
    for s in sizes:
        
        for t in thr:
            plt.figure(1)
            chunks=d_all[s][t][0]
            flops=d_all[s][t][1]
            
            plt.scatter(chunks,flops, label=str(t+1)+" threads")
            plt.xlabel('chunk size')
            plt.ylabel('MFLop/s')
            plt.xscale('log')
            plt.grid(True, 'both')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.title(date_str)
            ch_s=np.argsort(np.asarray(chunks))
            plt.savefig(pp, format='pdf',bbox_inches='tight')
            plt.figure(2)

            plt.plot([chunks[l] for l in ch_s], [flops[l] for l in ch_s], label=str(t+1)+" threads")
            plt.xlabel('chunk size')
            plt.ylabel('MFLop/s')
            plt.xscale('log')
            plt.title(date_str)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
    
    plt.show()
    pp.close()



#for t in thr:
#    a=[]
#    b=[]
##    plt.figure(t+1)
#    for k in d_all[1000000].keys():
#        if 'openmp' not in k:
#            a.append(d_all[1000000][k][t])
#            b.append(int(k.split('-')[0]))
#            
#    plt.scatter(b,a, label=str(t+1)+" threads")
#    plt.xlabel('chunk size')
#    plt.ylabel('MFLop/s')
#    plt.xscale('log')
#
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    
#    
#for t in thr:
#    a=[]
#    b=[]
##    plt.figure(t+1)
#    for k in d_all[1000000].keys():
#        if 'openmp' not in k:
#            a.append(d_all[1000000][k][t])
#            b.append(int(k.split('-')[0]))
#    indices=np.argsort(b)  
#      
#    plt.plot([b[i] for i in indices],[a[i] for i in indices], label=str(t+1)+" threads")
#    plt.xlabel('chunk size')
#    plt.ylabel('MFLop/s')
#    plt.xscale('log')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)