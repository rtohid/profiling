#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:28:13 2018

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")

date_str='10-22-18-1554'
openmp_dir='/home/shahrzad/Spyder/Blazemark/results/data/openmp/'+date_str
from matplotlib.backends.backend_pdf import PdfPages

    
thr=[]
sizes=[]
repeats=[]

data_files=glob.glob(openmp_dir+'/*.dat')
d_openmp={}

openmp_benchmarks=[]
d_openmp_all={}   
for filename in data_files:
    if 'openmp' in filename:
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in openmp_benchmarks:
            openmp_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
openmp_benchmarks.sort()      
repeats.sort()      

sizes=[100, 1000,10000, 100000, 1000000, 10000000]

for repeat in repeats:
    d_openmp_all[repeat]={}
    for benchmark in openmp_benchmarks:   
        d_openmp_all[repeat][benchmark]={}
        for s in sizes:
            d_openmp_all[repeat][benchmark][s]=[0]*len(thr)

for benchmark in openmp_benchmarks:   
        d_openmp[benchmark]={}
        for s in sizes:
            d_openmp[benchmark][s]=[0]*len(thr)            

data_files.sort()        
for filename in data_files:
    if 'openmp' in filename:
        (repeat,benchmark, cores, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        print(repeat, benchmark, cores, runtime)    
    

    f=open(filename, 'r')
             
    if 'openmp' in filename:        
        f=open(filename, 'r')
        result=f.readlines()[3:9]

        for r in result:
            s=int(r.strip().split(' ')[0])
            print("s:",s)
            if s in sizes:
                d_openmp_all[int(repeat)][benchmark][s][int(cores)-1]=float(r.strip().split(' ')[-1]) 

pp = PdfPages(openmp_dir+'/blazemark_openmp_performance.pdf')

i=1
for benchmark in openmp_benchmarks:   
    for s in sizes:
        for repeat in repeats[1:]:
            for th in thr:
                d_openmp[benchmark][s][th-1]+=d_openmp_all[repeat][benchmark][s][th-1]
        d_openmp[benchmark][s]=[d_openmp[benchmark][s][th-1]/max(repeats) for th in thr]
        plt.figure(i)
    
        plt.plot(thr, d_openmp[benchmark][s],label='openmp', linestyle='--', color='black')
        plt.xlabel('#number of cores')
        plt.ylabel('MFLop/s')
        plt.title(benchmark+'  '+date_str+'\n\n\nvector size: '+str(s))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        i=i+1
        
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        
plt.show()
pp.close() 
        
import pickle
#################################
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
##################################       
        
save_obj(d_openmp, openmp_dir+'/openmp_benchmarks')
