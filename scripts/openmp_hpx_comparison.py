#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:23:21 2018

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")


openmp_date_str='11-03-18-1413'
hpx_date_str='11-03-18-1413'
hpx_dir='/home/shahrzad/repos/hpx_test/data/'+hpx_date_str
hpx_perf_directory='/home/shahrzad/repos/hpx_test/plots/'+hpx_date_str

openmp_dir='/home/shahrzad/repos/openmp_test/data/'+openmp_date_str
openmp_perf_directory='/home/shahrzad/repos/openmp_test/plots/'+openmp_date_str

from matplotlib.backends.backend_pdf import PdfPages

    
data_files=glob.glob(hpx_dir+'/*.dat')
d_hpx={}
thr=[]
sizes=[]
repeats=[]

for filename in data_files:
    if 'date' not in filename:
        repeat = int(filename.split('/')[-1].replace('.dat','').split('_')[0])
        th = int(filename.split('/')[-1].replace('.dat','').split('_')[-1])
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
repeats.sort()      

d_hpx_all={}   

for repeat in repeats:
    d_hpx_all[repeat]={}
    for th in thr:
        d_hpx_all[repeat][th]={}

     
data_files.sort()        
for filename in data_files:
    if 'date' not in filename:
        size=[]
        mflops=[]    
    
        f=open(filename, 'r')
                 
        result=f.readlines()
        for i in range(len(result)):
            if 'vector size: ' in result[i]:
                size.append(int(result[i].strip().split(': ')[1]))
            if 'MFLOPS: ' in result[i]:
#                if result[i].strip().split(': ')[1] == 'inf':
#                    mflops.append(1e5)
#                else:
                mflops.append(float(result[i].strip().split(': ')[1]))    
                
        repeat = int(filename.split('/')[-1].replace('.dat','').split('_')[0])
        th = int(filename.split('/')[-1].replace('.dat','').split('_')[-1])
            
        d_hpx_all[repeat][th]['size']=size
        d_hpx_all[repeat][th]['mflops']=mflops
 
d_hpx={}
for th in thr[0:8]:
    d_hpx[th]={}
    size_0=d_hpx_all[1][th]['size']
    mflops=[0]*len(size_0)
    for r in repeats[1:]:
        size=d_hpx_all[r][th]['size']
        if size!=size_0:
            print("errorrrrrrrrrrrrrrrrrrrrr")
        mflops=[mflops[i]+d_hpx_all[r][th]['mflops'][i] for i in range(len(mflops))]
    d_hpx[th]['size']=size_0
    d_hpx[th]['mflops']=[x/float(max(repeats)-1) for x in mflops]


####################################################
#opemp
####################################################        
data_files=glob.glob(openmp_dir+'/*.dat')
d_openmp={}
thr=[]
sizes=[]
repeats=[]

openmp_benchmarks=[]
for filename in data_files:
    if 'date' not in filename:
        repeat = int(filename.split('/')[-1].replace('.dat','').split('_')[0])
        th = int(filename.split('/')[-1].replace('.dat','').split('_')[-1])
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
repeats.sort()      

d_openmp_all={}   

for repeat in repeats:
    d_openmp_all[repeat]={}
    for th in thr:
        d_openmp_all[repeat][th]={}

     
data_files.sort()        
for filename in data_files:
    if 'date' not in filename:
        size=[]
        mflops=[]    
    
        f=open(filename, 'r')
                 
        result=f.readlines()
        for i in range(len(result)):
            if i%4==0:
                size.append(int(result[i].strip().split(': ')[1]))
            if i%4==2:
                mflops.append(float(result[i].strip().split(': ')[1]))    
                
        repeat = int(filename.split('/')[-1].replace('.dat','').split('_')[0])
        th = int(filename.split('/')[-1].replace('.dat','').split('_')[-1])
            
        d_openmp_all[repeat][th]['size']=size
        d_openmp_all[repeat][th]['mflops']=mflops
 
d_openmp={}
for th in thr:
    d_openmp[th]={}
    size_0=d_openmp_all[1][th]['size']
    mflops=[0]*len(size_0)
    for r in repeats[1:]:
        size=d_openmp_all[r][th]['size']
        if size!=size_0:
            print("errorrrrrrrrrrrrrrrrrrrrr")
        mflops=[mflops[i]+d_openmp_all[r][th]['mflops'][i] for i in range(len(mflops))]
    d_openmp[th]['size']=size_0
    d_openmp[th]['mflops']=[x/float(max(repeats)-1) for x in mflops]

pp = PdfPages(openmp_perf_directory+'/perf_log_openmp_'+date_str+'.pdf')

i=1
plt.figure(i)
for th in thr:
    plt.plot(d_openmp[th]['size'], d_openmp[th]['mflops'],label=str(th)+' threads')
    plt.xlabel("# vector size")           
    plt.ylabel('MFlops')
    plt.xscale('log')
    plt.grid(True, 'both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('openmp  '+openmp_date_str)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(pp, format='pdf',bbox_inches='tight')
print('')
plt.show()
pp.close() 

pp = PdfPages(hpx_perf_directory+'/perf_log_openmp_'+date_str+'.pdf')

i=i+1
plt.figure(i)
for th in thr:
    plt.plot(d_hpx[th]['size'], d_hpx[th]['mflops'],label=str(th)+' threads')
    plt.xlabel("# vector size")           
    plt.ylabel('MFlops')
    plt.xscale('log')
    plt.grid(True, 'both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('hpx   '+hpx_date_str)
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
