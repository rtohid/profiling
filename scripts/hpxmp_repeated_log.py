#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:37:23 2018

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
#
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")
benchmark='dvecdvecadd'
#benchmark='daxpy'
#benchmark='dmatdmatadd'
#benchmark='dmatdmatmult'
#openmp_date_str='12-14-2018-1512'
#openmp_date_str='01-17-2019-1242'
#openmp_date_str='11-19-18-0936' #daxpy
#openmp_date_str='12-14-2018-1512' #dmatdmatmult
openmp_date_str='01-17-2019-1242' #dmatdmatadd

openmp_dir='/home/shahrzad/repos/Blazemark/data/openmp/all'

hpxmp_dir_1='/home/shahrzad/repos/Blazemark/data/hpxmp/all'
hpxmp_dir_2='/home/shahrzad/repos/Blazemark/data/hpxmp/2/idle_on'
hpxmp_dir_3='/home/shahrzad/repos/Blazemark/data/hpxmp/3'
hpxmp_dir_4='/home/shahrzad/repos/Blazemark/data/hpxmp/4'

hpx_ref_date_str='11-22-18-1027'  #reference hpx dvecdvecadd
hpx_ref_dir='/home/shahrzad/repos/Blazemark/data/vector/'+hpx_ref_date_str
hpx_counters_dir='/home/shahrzad/repos/Blazemark/data/hpxmp/3/dvecdvecadd_6repeat_201216size/hpx'
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/hpxmp/3'

from matplotlib.backends.backend_pdf import PdfPages

#####################################################
                        #hpx refernce
#####################################################
thr=[]
sizes=[]
repeats=[]
block_sizes=[]
chunk_sizes=[]
block_sizes_row=[]
block_sizes_col=[]

data_files=glob.glob(hpx_ref_dir+'/*.dat')

hpx_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:
        if len(filename.split('/')[-1].replace('.dat','').split('-'))==6:
            option=1
            (repeat, benchmark, th, runtime, chunk_size, block_size) = filename.split('/')[-1].replace('.dat','').split('-')
            if benchmark not in hpx_benchmarks:
                hpx_benchmarks.append(benchmark)
                
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat))
            if int(block_size) not in block_sizes:
                block_sizes.append(int(block_size))
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
        elif len(filename.split('/')[-1].replace('.dat','').split('-'))==7:
            option=2
            (repeat, benchmark, th, runtime, chunk_size, block_size_row, block_size_col) = filename.split('/')[-1].replace('.dat','').split('-')
            if benchmark not in hpx_benchmarks:
                hpx_benchmarks.append(benchmark)
                
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat))
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
            if str(block_size_row)+'-'+str(block_size_col) not in block_sizes:
                block_sizes.append(str(block_size_row)+'-'+str(block_size_col))                
        
thr.sort()
hpx_benchmarks.sort()      
repeats.sort()      
block_sizes.sort()
chunk_sizes.sort()

d_hpx_all={}   
d_hpx={}
for benchmark in hpx_benchmarks:  
    d_hpx_all[benchmark]={}
    d_hpx[benchmark]={}
    for c in chunk_sizes:
        d_hpx_all[benchmark][c]={}
        d_hpx[benchmark][c]={}
        for b in block_sizes:
            d_hpx_all[benchmark][c][b]={}
            d_hpx[benchmark][c][b]={}
            for th in thr:
                d_hpx_all[benchmark][c][b][th]={}
                d_hpx[benchmark][c][b][th]={}
                for r in repeats:
                    d_hpx_all[benchmark][c][b][th][r]={}        
                                        
data_files.sort()        
for filename in data_files:    
    if 'hpx' in filename:
        
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        chunk=int(filename.split('/')[-1].split('-')[4]) 
        if option ==1:
            block=int(filename.split('/')[-1].split('-')[5][0:-4])       
        elif option ==2:
            block=filename.split('/')[-1].split('-')[5]+'-'+filename.split('/')[-1].split('-')[6][0:-4]     

        if block in block_sizes:  
            size=[]
            mflops=[]    
            for r in result:        
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
            
            d_hpx_all[benchmark][chunk][block][th][repeat]['size']=size
            d_hpx_all[benchmark][chunk][block][th][repeat]['mflops']=mflops
 
    
for benchmark in hpx_benchmarks:
    for c in chunk_sizes:
        for b in block_sizes:
            for th in thr:
                mflops=[0]*len(size)

                if max(repeats)==1:
                    if 'mflops' in d_hpx_all[benchmark][c][b][th][1].keys():
                        d_hpx[benchmark][c][b][th]['size']=size
                        mflops=d_hpx_all[benchmark][c][b][th][repeats[0]]['mflops']
                        d_hpx[benchmark][c][b][th]['mflops']=mflops
                else:
                    d_hpx[benchmark][c][b][th]['size']=size
                    for r in repeats[1:]:
                        mflops=[mflops[i]+d_hpx_all[benchmark][c][b][th][r]['mflops'][i] for i in range(len(mflops))]                        
                        d_hpx[benchmark][c][b][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
       
        
####################################################
#openmp
####################################################        
data_files=glob.glob(openmp_dir+'/*.dat')
d_openmp={}
thr=[]
repeats=[]
openmp_benchmarks=[]
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

d_openmp_all={}   

for repeat in repeats:
    d_openmp_all[repeat]={}
    for benchmark in openmp_benchmarks:   
        d_openmp_all[repeat][benchmark]={}
        for th in thr:
            d_openmp_all[repeat][benchmark][th]={}

     

data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    

    if 'openmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
    
            
        for r in result:
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
        
        d_openmp_all[repeat][benchmark][th]['size']=size
        d_openmp_all[repeat][benchmark][th]['mflops']=mflops
 
d_openmp={}
for benchmark in openmp_benchmarks:
    d_openmp[benchmark]={}
    for th in thr:
        d_openmp[benchmark][th]={}
        size_0=d_openmp_all[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        for r in repeats[1:]:
            size=d_openmp_all[r][benchmark][th]['size']
            if size!=size_0:
                print("errorrrrrrrrrrrrrrrrrrrrr")
            mflops=[mflops[i]+d_openmp_all[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]
        d_openmp[benchmark][th]['size']=size_0
        d_openmp[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
        
####################################################
#hpxmp_1
####################################################        
data_files=glob.glob(hpxmp_dir_1+'/*.dat')
d_hpxmp_1={}

hpxmp_benchmarks_1=[]
for filename in data_files:
    if 'openmp' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpxmp_benchmarks_1:
            hpxmp_benchmarks_1.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
hpxmp_benchmarks_1.sort()      
repeats.sort()      

d_hpxmp_all_1={}   

for repeat in repeats:
    d_hpxmp_all_1[repeat]={}
    for benchmark in hpxmp_benchmarks_1:   
        d_hpxmp_all_1[repeat][benchmark]={}
        for th in thr:
            d_hpxmp_all_1[repeat][benchmark][th]={}

     

data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    

    if 'openmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
    
            
        for r in result:
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
        
        d_hpxmp_all_1[repeat][benchmark][th]['size']=size
        d_hpxmp_all_1[repeat][benchmark][th]['mflops']=mflops
 
d_hpxmp_1={}
for benchmark in hpxmp_benchmarks_1:
    d_hpxmp_1[benchmark]={}
    for th in thr:
        d_hpxmp_1[benchmark][th]={}
        size_0=d_hpxmp_all_1[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        for r in repeats[1:]:
            size=d_hpxmp_all_1[r][benchmark][th]['size']
            if size!=size_0:
                print(str(th)+"errorrrrrrrrrrrrrrrrrrrrr")
            mflops=[mflops[i]+d_hpxmp_all_1[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]
        d_hpxmp_1[benchmark][th]['size']=size_0
        d_hpxmp_1[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
                
        
####################################################
#hpxmp_2
####################################################        
data_files=glob.glob(hpxmp_dir_2+'/*.dat')
d_hpxmp_2={}

thr=[]
repeats=[]
hpxmp_benchmarks_2=[]
for filename in data_files:
    if 'hpxmp' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpxmp_benchmarks_2:
            hpxmp_benchmarks_2.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
hpxmp_benchmarks_2.sort()      
repeats.sort()      
d_hpxmp_all_2={}   

for repeat in repeats:
    d_hpxmp_all_2[repeat]={}
    for benchmark in hpxmp_benchmarks_2:   
        d_hpxmp_all_2[repeat][benchmark]={}
        for th in thr:
            d_hpxmp_all_2[repeat][benchmark][th]={}

     

data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    

    if 'hpxmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        if th in thr:
            
            for r in result:
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
            
            d_hpxmp_all_2[repeat][benchmark][th]['size']=size
            d_hpxmp_all_2[repeat][benchmark][th]['mflops']=mflops
     
d_hpxmp_2={}
for benchmark in hpxmp_benchmarks_2:
    d_hpxmp_2[benchmark]={}
    for th in thr:
        d_hpxmp_2[benchmark][th]={}
        size_0=d_hpxmp_all_2[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        for r in repeats[1:]:
            size=d_hpxmp_all_2[r][benchmark][th]['size']
            if size!=size_0:
                print(str(th)+"errorrrrrrrrrrrrrrrrrrrrr")
            mflops=[mflops[i]+d_hpxmp_all_2[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]
        d_hpxmp_2[benchmark][th]['size']=size_0
        d_hpxmp_2[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]

####################################################
#hpxmp_3
####################################################        
data_files=glob.glob(hpxmp_dir_3+'/*.dat')
d_hpxmp_3={}

thr=[]
repeats=[]
hpxmp_benchmarks_3=[]
for filename in data_files:
    if 'hpxmp' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpxmp_benchmarks_3:
            hpxmp_benchmarks_3.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
hpxmp_benchmarks_3.sort()      
repeats.sort()      
d_hpxmp_all_3={}   

for repeat in repeats:
    d_hpxmp_all_3[repeat]={}
    for benchmark in hpxmp_benchmarks_3:   
        d_hpxmp_all_3[repeat][benchmark]={}
        for th in thr:
            d_hpxmp_all_3[repeat][benchmark][th]={}


data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    

    if 'hpxmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        if th in thr:
            
            for r in result:
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
            
            d_hpxmp_all_3[repeat][benchmark][th]['size']=size
            d_hpxmp_all_3[repeat][benchmark][th]['mflops']=mflops
     
d_hpxmp_3={}
for benchmark in hpxmp_benchmarks_3:
    d_hpxmp_3[benchmark]={}
    for th in thr:
        d_hpxmp_3[benchmark][th]={}
        size_0=d_hpxmp_all_3[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        size=d_hpxmp_all_3[1][benchmark][th]['size']
        mflops=d_hpxmp_all_3[1][benchmark][th]['mflops']
        d_hpxmp_3[benchmark][th]['size']=size
        d_hpxmp_3[benchmark][th]['mflops']=mflops
                                
        #        for r in repeats[1:]:
#            size=d_hpxmp_all_3[r][benchmark][th]['size']
#            if size!=size_0:
#                print(str(th)+"errorrrrrrrrrrrrrrrrrrrrr")
#            mflops=[mflops[i]+d_hpxmp_all_3[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]
#        d_hpxmp_3[benchmark][th]['size']=size_0
#        d_hpxmp_3[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
####################################################
#hpxmp_4
####################################################        
data_files=glob.glob(hpxmp_dir_4+'/*.dat')
d_hpxmp_4={}

thr=[]
repeats=[]
hpxmp_benchmarks_4=[]
for filename in data_files:
    if 'hpxmp' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpxmp_benchmarks_4:
            hpxmp_benchmarks_4.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
hpxmp_benchmarks_4.sort()      
repeats.sort()      
d_hpxmp_all_4={}   

for repeat in repeats:
    d_hpxmp_all_4[repeat]={}
    for benchmark in hpxmp_benchmarks_4:   
        d_hpxmp_all_4[repeat][benchmark]={}
        for th in thr:
            d_hpxmp_all_4[repeat][benchmark][th]={}


data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    

    if 'hpxmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        if th in thr:
            
            for r in result:
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
            
            d_hpxmp_all_4[repeat][benchmark][th]['size']=size
            d_hpxmp_all_4[repeat][benchmark][th]['mflops']=mflops
     
d_hpxmp_4={}
for benchmark in hpxmp_benchmarks_4:
    d_hpxmp_4[benchmark]={}
    for th in thr:
        d_hpxmp_4[benchmark][th]={}
        size_0=d_hpxmp_all_4[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        size=d_hpxmp_all_4[1][benchmark][th]['size']
        mflops=d_hpxmp_all_4[1][benchmark][th]['mflops']
        d_hpxmp_4[benchmark][th]['size']=size
        d_hpxmp_4[benchmark][th]['mflops']=mflops
        
####################################################
#hpx performance counters
####################################################        
data_files=glob.glob(hpx_counters_dir+'/*.dat')
d_hpxmp_counters={}

thr=[]
repeats=[]
hpxmp_counters_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpxmp_counters_benchmarks:
            hpxmp_counters_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
hpxmp_counters_benchmarks.sort()      
repeats.sort()      
d_hpxmp_counters_all={}   

for repeat in repeats:
    d_hpxmp_counters_all[repeat]={}
    for benchmark in hpxmp_counters_benchmarks:   
        d_hpxmp_counters_all[repeat][benchmark]={}
        for th in thr:
            d_hpxmp_counters_all[repeat][benchmark][th]={}


data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    
    if 'hpxmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        if th in thr:
            counters={'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th}

            for r in result:
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
                elif "threads" in r:
                    if 'idle-rate' in r and 'pool' in r:
                        idle_rate=float(r.strip().split(',')[-2])/100
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['idle_rate'][th_num]=idle_rate
                    if 'average,' in r and 'pool' in r:
                        average_time=float(r.strip().split(',')[-2])/1000
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['average_time'][th_num]=average_time
                    if 'cumulative,' in r and 'pool' in r:
                        cumulative=float(r.strip().split(',')[-1])
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['cumulative_count'][th_num]=cumulative
                    if 'cumulative-overhead' in r and 'pool' in r:
                        cumulative_overhead=float(r.strip().split(',')[-2])/1000
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['cumulative_overhead_time'][th_num]=cumulative_overhead
                    if 'average-overhead' in r and 'pool' in r:
                        average_overhead=float(r.strip().split(',')[-2])/1000
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['average_overhead_time'][th_num]=average_overhead                        
            d_hpxmp_counters_all[repeat][benchmark][th]['size']=size
            d_hpxmp_counters_all[repeat][benchmark][th]['mflops']=mflops
            d_hpxmp_counters_all[repeat][benchmark][th]['counters']=counters
            
d_hpxmp_counters={}
for benchmark in hpxmp_counters_benchmarks:
    d_hpxmp_counters[benchmark]={}
    for th in thr:        
        d_hpxmp_counters[benchmark][th]={}
        size_0=d_hpxmp_counters_all[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        counter_d_0=[0]*th
        counter_d_1=[0]*th
        counter_d_2=[0]*th
        counter_d_3=[0]*th
        counter_d_4=[0]*th
        d_hpxmp_counters[benchmark][th]['counters']={}
        for r in repeats[1:]:        
            size=d_hpxmp_counters_all[r][benchmark][th]['size']
            mflops=[mflops[i]+d_hpxmp_counters_all[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]                
            counter_d_0=[counter_d_0[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['idle_rate'][i] for i in range(th)]
            counter_d_1=[counter_d_1[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['average_time'][i] for i in range(th)]
            counter_d_2=[counter_d_2[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['cumulative_overhead_time'][i] for i in range(th)]
            counter_d_3=[counter_d_3[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['cumulative_count'][i] for i in range(th)]
            counter_d_4=[counter_d_4[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['average_overhead_time'][i] for i in range(th)]

            
        d_hpxmp_counters[benchmark][th]['size']=size_0
        d_hpxmp_counters[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
        d_hpxmp_counters[benchmark][th]['counters']['idle_rate']=[x/float(max(repeats)-1) for x in counter_d_0]
        d_hpxmp_counters[benchmark][th]['counters']['average_time']=[x/float(max(repeats)-1) for x in counter_d_1]
        d_hpxmp_counters[benchmark][th]['counters']['cumulative_overhead_time']=[x/float(max(repeats)-1) for x in counter_d_2]
        d_hpxmp_counters[benchmark][th]['counters']['cumulative_count']=[x/float(max(repeats)-1) for x in counter_d_3]
        d_hpxmp_counters[benchmark][th]['counters']['average_overhead_time']=[x/float(max(repeats)-1) for x in counter_d_4]
i=1
for benchmark in hpxmp_counters_benchmarks:  
    plt.figure(i)
    j=d_openmp[benchmark][th]['size'].index(size[0])
    pp = PdfPages(perf_directory+'/hpx_performance.pdf')
    plt.plot(thr, [d_openmp[benchmark][th]['mflops'][j] for th in thr], label='openmp')
    plt.plot(thr, [d_hpxmp_counters[benchmark][th]['mflops'][0] for th in thr], label='hpx')
    plt.xlabel("#threads")           
    plt.ylabel('MFlops')
    plt.grid(True, 'both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(benchmark+ " vector size 201,216")
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    plt.show()
    pp.close() 
    i=i+1
    j=d_openmp[benchmark][th]['size'].index(size[0])
    for th in thr:
        pp = PdfPages(perf_directory+'/hpx_performance_counters_'+str(th)+'.pdf')

        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['idle_rate'], label='idle_rate')
        plt.xlabel("#threads")      
        plt.xticks(np.arange(1,th+1).tolist())
        plt.ylabel('%')
        plt.title('idle_rate')
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        i=i+1
        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['average_time'], label='average_time')
        plt.xlabel("#threads")     
        plt.xticks(np.arange(1,th+1).tolist())
        plt.ylabel('Microseconds')
        plt.title('average_time')
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        i=i+1
        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['cumulative_overhead_time'], label='cumulative_overhead_time')
        plt.xlabel("#threads") 
        plt.xticks(np.arange(1,th+1).tolist())                   
        plt.ylabel('Microseconds')
        plt.title('cumulative_overhead_time')
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        i=i+1
        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['cumulative_count'], label='cumulative_count')
        plt.xlabel("#threads") 
        plt.xticks(np.arange(1,th+1).tolist())                   
#        plt.ylabel('')
        plt.title("cumulative_count")
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        i=i+1
        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['average_overhead_time'], label='average_overhead_time')
        plt.xlabel("#threads")    
        plt.xticks(np.arange(1,th+1).tolist())                   
        plt.ylabel('Microseconds')
        plt.title("average_overhead_time")
        i=i+1
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        plt.show()
        pp.close() 
    
    
        
##############################################################################
        #plots
##############################################################################
#pp = PdfPages(perf_directory+'/hpxmp_idle.pdf')
i=1
for benchmark in hpxmp_benchmarks_2:       
    for th in [1, 4,8,16]:
#        pp = PdfPages(perf_directory+'/figures/'+benchmark+'_'+str(th)+'.pdf')

        plt.figure(i)

        plt.plot(d_hpxmp_1[benchmark][th]['size'], d_hpxmp_1[benchmark][th]['mflops'],label="hpxMP previous",color='black',linestyle=':')
#        if benchmark in d_hpxmp_2.keys():
#            plt.plot(d_hpxmp_2[benchmark][th]['size'], d_hpxmp_2[benchmark][th]['mflops'],label='hpxmp_idle_on '+str(th)+' threads')
        plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label="llvm-OpenMP", color='black')

#        plt.plot(d_hpxmp_2[benchmark][th]['size'], d_hpxmp_2[benchmark][th]['mflops'],label="hpxMP", color='black', linestyle='dashed')
        plt.plot(d_hpxmp_4[benchmark][th]['size'], d_hpxmp_4[benchmark][th]['mflops'],label="hpxMP", color='black', linestyle='dashed')

#        plt.plot(d_hpxmp_3[benchmark][th]['size'], d_hpxmp_3[benchmark][th]['mflops'],label="hpxmp "+str(th)+" threads", color='black', linestyle='dashed')
#        plt.plot(d_hpx[benchmark][10][256][th]['size'], d_hpx[benchmark][10][256][th]['mflops'],label='hpx '+str(th)+' threads')

        plt.xlabel("size $n$")           
        plt.ylabel('MFlops ('+str(th)+" threads)")
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.legend(loc=2)
#        plt.title(benchmark)
        i=i+1
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        plt.show()
        pp.close() 
        
        
        
        
        
i=1
for benchmark in benchmarks:       
    for th in [1, 4,8,16]:
        pp = PdfPages(perf_directory+'/'+benchmark+'_'+str(th)+'.pdf')

        plt.figure(i)
        plt.plot(d_hpxmp_1[benchmark][th]['size'], d_hpxmp_1[benchmark][th]['mflops'],label="hpxMP previous",color='black',linestyle='dashed')

        plt.plot(d_hpxmp[node][benchmark][th]['size'], d_hpxmp[node][benchmark][th]['mflops'],label="hpxMP",color='black',linestyle='solid')
#        if benchmark in d_hpxmp_2.keys():
#            plt.plot(d_hpxmp_2[benchmark][th]['size'], d_hpxmp_2[benchmark][th]['mflops'],label='hpxmp_idle_on '+str(th)+' threads')
        plt.plot(d_openmp[node][benchmark][th]['size'], d_openmp[node][benchmark][th]['mflops'],label="OpenMP", color='gray', linestyle='solid')

#        plt.plot(d_hpxmp_3[benchmark][th]['size'], d_hpxmp_3[benchmark][th]['mflops'],label="hpxmp "+str(th)+" threads", color='black', linestyle='dashed')
#        plt.plot(d_hpx[benchmark][10][256][th]['size'], d_hpx[benchmark][10][256][th]['mflops'],label='hpx '+str(th)+' threads')

        plt.xlabel("size $n$")           
        plt.ylabel('MFlops ('+str(th)+" threads)")
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.legend(loc=2)
#        plt.title(benchmark)
        i=i+1
        
        
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        plt.show()
        pp.close()         
        
        
        
        
        
        
        
        
        
#    print('')
#plt.show()
#pp.close() 
        
#for th in thr:
#    a=(d_hpxmp_2['dvecdvecadd'][th]['mflops'][135]-d_hpxmp_1['dvecdvecadd'][th]['mflops'][135])/d_hpxmp_1['dvecdvecadd'][th]['mflops'][135]
#    print(str(th)+': '+str(a))

#th=16     
#s=0   
#for i in range(1,12):
#    print(d_hpxmp_all_1[i]['dvecdvecadd'][th]['mflops'][45])
#    print(d_hpxmp_all_2[i]['dvecdvecadd'][th]['mflops'][45])
#    print(d_hpxmp_all_3[i]['dvecdvecadd'][th]['mflops'][45])
#
#    print('----')
#    
#for th in thr:    
#    s=[0]*len(d_hpxmp_all_1[1]['dvecdvecadd'][th]['mflops'])
#    for r in repeats[1:]:
#        s= [s[i]+d_hpxmp_all_1[r]['dvecdvecadd'][th]['mflops'][i] for i in range(len(mflops))]
#    s=[s[i]/(len(repeats)-1) for i in range(len(mflops))]
#    v=[0]*len(d_hpxmp_all_1[1]['dvecdvecadd'][th]['mflops'])
#    for r in repeats[1:]:
#        v= [v[i]+(d_hpxmp_all_1[r]['dvecdvecadd'][th]['mflops'][i]-s[i])*(d_hpxmp_all_1[r]['dvecdvecadd'][th]['mflops'][i]-s[i]) for i in range(len(mflops))] 
#    v=[np.sqrt(v[i])/((len(repeats)-1)*s[i]) for i in range(len(mflops))]
#    

