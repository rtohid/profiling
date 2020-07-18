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
sizes=[10000000]
if len(sizes)==1:
    directory='/home/shahrzad/Spyder/Blazemark/results/data/'+str(sizes[0])+'/2'
else:
    directory='/home/shahrzad/Spyder/Blazemark/results/data/'
data_files=glob.glob(directory+'/*.dat')

d_all={}
for s in sizes:
    d_all[s]={}

multiplyers=[]    
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
    f=open(filename, 'r')
    results=f.readlines()[3:9]

    for r in results:
        if len(sizes)==1:
            if '     '+str(sizes[0])+' ' in r:
                s=r.strip().split(' ')[0]
                print("s:",s)
                if k not in d_all[int(s)].keys():            
                    d_all[int(s)][k]=[0]*len(thr)
                d_all[int(s)][k][int(cores)-1]=float(r.strip().split(' ')[-1])                
        else:
                s=r.strip().split(' ')[0]
                print("s:",s)
                if k not in d_all[int(s)].keys():            
                    d_all[int(s)][k]=[0]*len(thr)
                d_all[int(s)][k][int(cores)-1]=float(r.strip().split(' ')[-1]) 

 
for m in multiplyers:
    pp = PdfPages(directory+'/blazemark_hpx_config_performance.pdf')

    i=1
    for s in sizes:
        plt.figure(i)
        for k in d_all[s]:
            if '-'+str(m)+'-1000' in k:
            #if '-' in k:
                plt.plot(thr, d_all[s][k],label="chunk size="+str(k.split('-')[0])+" multiplyer="+str(k.split('-')[1]))   
            #else:
            elif 'openmp' in k:
                plt.plot(thr, d_all[s][k],label='openmp', linestyle='--', color='black')
        plt.xlabel('#number of cores')
        plt.ylabel('MFLop/s')
        plt.title(date_str+'\n\n\nvector size: '+str(s)+'      threshold=1000')
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        i=i+1
        
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
    



plt.figure(i)
for s in sizes:
    for t in thr:
        a=[]
        b=[]
    #    plt.figure(t+1)
        for k in d_all[s].keys():
            if 'openmp' not in k:
                a.append(d_all[s][k][t])
                b.append(int(k.split('-')[0]))
                
        plt.scatter(b,a, label=str(t+1)+" threads")
        plt.xlabel('chunk size')
        plt.ylabel('MFLop/s')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.title(date_str)
    
        
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    
plt.show()
pp.close()
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
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)