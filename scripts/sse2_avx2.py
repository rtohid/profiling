#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:24:34 2019

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
from matplotlib.backends.backend_pdf import PdfPages

now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")
benchmark='dmatdmatadd'

#hpx_dir_sse2='/home/shahrzad/repos/Blazemark/data/vector/'+benchmark+'/sse2'
#hpx_dir_avx2='/home/shahrzad/repos/Blazemark/data/vector/'+benchmark+'/avx2'

hpx_dir_sse2='/home/shahrzad/repos/Blazemark/data/matrix/'+benchmark+'/trillian'
hpx_dir_avx2='/home/shahrzad/repos/Blazemark/data/matrix/'+benchmark+'/01-04-2019-1027'

########################################################################
#hpx_sse2
##########################################################################                        
thr=[]
sizes=[]
repeats=[]
block_sizes=[]
chunk_sizes=[]
block_sizes_row=[]
block_sizes_col=[]

data_files=glob.glob(hpx_dir_sse2+'/*.dat')

hpx_sse2_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:
        if len(filename.split('/')[-1].replace('.dat','').split('-'))==6:
            option=1
            (repeat, benchmark, th, runtime, chunk_size, block_size) = filename.split('/')[-1].replace('.dat','').split('-')
            if benchmark not in hpx_sse2_benchmarks:
                hpx_sse2_benchmarks.append(benchmark)
                
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
            if benchmark not in hpx_sse2_benchmarks:
                hpx_sse2_benchmarks.append(benchmark)
                
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat))
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
            if str(block_size_row)+'-'+str(block_size_col) not in block_sizes:
                block_sizes.append(str(block_size_row)+'-'+str(block_size_col))                
        
thr.sort()
hpx_sse2_benchmarks.sort()      
repeats.sort()      
block_sizes.sort()
chunk_sizes.sort()

d_hpx_sse2_all={}   
d_hpx_sse2={}
for benchmark in hpx_sse2_benchmarks:  
    d_hpx_sse2_all[benchmark]={}
    d_hpx_sse2[benchmark]={}
    for c in chunk_sizes:
        d_hpx_sse2_all[benchmark][c]={}
        d_hpx_sse2[benchmark][c]={}
        for b in block_sizes:
            d_hpx_sse2_all[benchmark][c][b]={}
            d_hpx_sse2[benchmark][c][b]={}
            for th in thr:
                d_hpx_sse2_all[benchmark][c][b][th]={}
                d_hpx_sse2[benchmark][c][b][th]={}
                for r in repeats:
                    d_hpx_sse2_all[benchmark][c][b][th][r]={}        
                                        
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
            
            d_hpx_sse2_all[benchmark][chunk][block][th][repeat]['size']=size
            d_hpx_sse2_all[benchmark][chunk][block][th][repeat]['mflops']=mflops
 
    
for benchmark in hpx_sse2_benchmarks:
    for c in chunk_sizes:
        for b in block_sizes:
            for th in thr:
                mflops=[0]*len(size)

                if max(repeats)==1:
                    if 'mflops' in d_hpx_sse2_all[benchmark][c][b][th][1].keys():
                        d_hpx_sse2[benchmark][c][b][th]['size']=size
                        mflops=d_hpx_sse2_all[benchmark][c][b][th][repeats[0]]['mflops']
                        d_hpx_sse2[benchmark][c][b][th]['mflops']=mflops
                else:
                    d_hpx_sse2[benchmark][c][b][th]['size']=size
                    for r in repeats[1:]:
                        mflops=[mflops[i]+d_hpx_sse2_all[benchmark][c][b][th][r]['mflops'][i] for i in range(len(mflops))]                        
                        d_hpx_sse2[benchmark][c][b][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
       
        
########################################################################
#hpx-avx2
##########################################################################                        
thr=[]
sizes=[]
repeats=[]
block_sizes=[]
chunk_sizes=[]
block_sizes_row=[]
block_sizes_col=[]

data_files=glob.glob(hpx_dir_avx2+'/*.dat')

hpx_avx2_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:
        if len(filename.split('/')[-1].replace('.dat','').split('-'))==6:
            option=1
            (repeat, benchmark, th, runtime, chunk_size, block_size) = filename.split('/')[-1].replace('.dat','').split('-')
            if benchmark not in hpx_avx2_benchmarks:
                hpx_avx2_benchmarks.append(benchmark)
                
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
            if benchmark not in hpx_avx2_benchmarks:
                hpx_avx2_benchmarks.append(benchmark)
                
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat))
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
            if str(block_size_row)+'-'+str(block_size_col) not in block_sizes:
                block_sizes.append(str(block_size_row)+'-'+str(block_size_col))                
        
thr.sort()
hpx_avx2_benchmarks.sort()      
repeats.sort()      
block_sizes.sort()
chunk_sizes.sort()

d_hpx_avx2_all={}   
d_hpx_avx2={}
for benchmark in hpx_avx2_benchmarks:  
    d_hpx_avx2_all[benchmark]={}
    d_hpx_avx2[benchmark]={}
    for c in chunk_sizes:
        d_hpx_avx2_all[benchmark][c]={}
        d_hpx_avx2[benchmark][c]={}
        for b in block_sizes:
            d_hpx_avx2_all[benchmark][c][b]={}
            d_hpx_avx2[benchmark][c][b]={}
            for th in thr:
                d_hpx_avx2_all[benchmark][c][b][th]={}
                d_hpx_avx2[benchmark][c][b][th]={}
                for r in repeats:
                    d_hpx_avx2_all[benchmark][c][b][th][r]={}        
                                        
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
            
            d_hpx_avx2_all[benchmark][chunk][block][th][repeat]['size']=size
            d_hpx_avx2_all[benchmark][chunk][block][th][repeat]['mflops']=mflops
 
    
for benchmark in hpx_avx2_benchmarks:
    for c in chunk_sizes:
        for b in block_sizes:
            for th in thr:
                mflops=[0]*len(size)

                if max(repeats)==1:
                    if 'mflops' in d_hpx_avx2_all[benchmark][c][b][th][1].keys():
                        d_hpx_avx2[benchmark][c][b][th]['size']=size
                        mflops=d_hpx_avx2_all[benchmark][c][b][th][repeats[0]]['mflops']
                        d_hpx_avx2[benchmark][c][b][th]['mflops']=mflops
                else:
                    d_hpx_avx2[benchmark][c][b][th]['size']=size
                    for r in repeats[1:]:
                        mflops=[mflops[i]+d_hpx_avx2_all[benchmark][c][b][th][r]['mflops'][i] for i in range(len(mflops))]                        
                        d_hpx_avx2[benchmark][c][b][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
       
 ###################################################################################       
    #plot  
 ###################################################################################       
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots'
        
pp = PdfPages(perf_directory+'/dmatdmataddd_trillianz_blocksize.pdf')
c=10
i=1
for th in thr:
    for b in d_hpx_avx2[benchmark][c].keys():
        if b in d_hpx_sse2[benchmark][c].keys() and ('size' in d_hpx_avx2[benchmark][c][b][th]):
            plt.figure(i)
            plt.plot(d_hpx_avx2[benchmark][c][b][th]['size'], d_hpx_avx2[benchmark][c][b][th]['mflops'],label='marvin2- chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
        
            plt.plot(d_hpx_sse2[benchmark][c][b][th]['size'], d_hpx_sse2[benchmark][c][b][th]['mflops'],label='trillian- chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
        
            plt.xlabel("# matrix size")           
            plt.ylabel('MFlops')
            plt.xscale('log')
            plt.grid(True, 'both')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.title(benchmark)
            i=i+1
            plt.savefig(pp, format='pdf',bbox_inches='tight')
            print('')
    plt.show()
    pp.close() 


c=10
i=1
for th in thr:
    for a in [1, 4, 8, 16]:
        for b in d_hpx_sse2[benchmark][c].keys():
            if b.startswith(str(a)):
                plt.figure(i)
                plt.plot(d_hpx_sse2[benchmark][c][b][th]['size'], d_hpx_sse2[benchmark][c][b][th]['mflops'],label='trillian- chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
            
                plt.xlabel("# matrix size")           
                plt.ylabel('MFlops')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.title(benchmark)
        i=i+1
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
    plt.show()
    pp.close() 