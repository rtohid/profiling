#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:54:39 2019

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

#hpx_dir='/home/shahrzad/repos/Blazemark/data/vector/'+benchmark+'/sse2'
#hpx_dir_avx2='/home/shahrzad/repos/Blazemark/data/vector/'+benchmark+'/avx2'

hpx_dir='/home/shahrzad/repos/Blazemark/data/matrix/dmatdmatadd/cache_miss'

########################################################################
#hpx
##########################################################################                        
thr=[]
repeats=[]
block_sizes=[]
chunk_sizes=[]
block_sizes_row=[]
block_sizes_col=[]
mat_sizes=[]
data_files=glob.glob(hpx_dir+'/*.dat')

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
        elif len(filename.split('/')[-1].replace('.dat','').split('-'))==8:
            option=3
            (repeat, benchmark, th, runtime, chunk_size, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')
            mat_size=mat_size.split(',')[0]
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
            if int(mat_size) not in mat_sizes:
                mat_sizes.append(int(mat_size))
        
thr.sort()
hpx_benchmarks.sort()      
repeats.sort()      
block_sizes.sort()
chunk_sizes.sort()
mat_sizes.sort()

d_hpx_all={}   
d_hpx={}
d_cache={}

for benchmark in hpx_benchmarks:  
    d_hpx_all[benchmark]={}
    d_hpx[benchmark]={}
    d_cache[benchmark]={}
    for m in mat_sizes:
        d_hpx_all[benchmark][m]={}
        d_hpx[benchmark][m]={}
        d_cache[benchmark][m]={}
        for c in chunk_sizes:
            d_hpx_all[benchmark][m][c]={}
            d_hpx[benchmark][m][c]={}
            d_cache[benchmark][m][c]={}
            for b in block_sizes:
                d_hpx_all[benchmark][m][c][b]={}
                d_hpx[benchmark][m][c][b]={}
                d_cache[benchmark][m][c][b]={}
                for th in thr:
                    d_hpx_all[benchmark][m][c][b][th]={}
                    d_hpx[benchmark][m][c][b][th]={}
                    d_cache[benchmark][m][c][b][th]={}
                    for r in repeats:
                        d_hpx_all[benchmark][m][c][b][th][r]={}        
                                        
data_files.sort()        
for filename in data_files:    
    if 'hpx' in filename:
        
        stop=False
        f=open(filename, 'r')
                 
        r=f.readlines()[3]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        chunk=int(filename.split('/')[-1].split('-')[4]) 
        if option ==1:
            block=int(filename.split('/')[-1].split('-')[5][0:-4])       
        elif option ==2:
            block=filename.split('/')[-1].split('-')[5]+'-'+filename.split('/')[-1].split('-')[6][0:-4]     
        elif option ==3:
            block=filename.split('/')[-1].split('-')[5]+'-'+filename.split('/')[-1].split('-')[6] 

        if block in block_sizes:  
            mat_size=(int(r.strip().split(' ')[0]))
            mflops=(float(r.strip().split(' ')[-1]))
            
            d_hpx_all[benchmark][mat_size][chunk][block][th][repeat]=mflops
 
    
for benchmark in hpx_benchmarks:
    for m in mat_sizes:
        for c in chunk_sizes:
            for b in block_sizes:
                for th in thr:    
                    if max(repeats)==1:
                        mflops=d_hpx_all[benchmark][m][c][b][th][repeats[0]]
                        d_hpx[benchmark][m][c][b][th]=mflops
                    else:
                        for r in repeats[1:]:
                            mflops=[mflops+d_hpx_all[benchmark][m][c][b][th][r] for r in repeats]                        
                            d_hpx[benchmark][m][c][b][th]=mflops/float(max(repeats)-1)


 ###################################################################################       
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots'
        
pp = PdfPages(perf_directory+'/dmatdmataddd_cache_miss_block_sizes.pdf')
       
i=1
for th in thr:
    for c in chunk_sizes:
        for a in [1, 4, 8]:
            for b in block_sizes:
                if b.startswith(str(a)):
                    p=[]
                    for m in mat_sizes:                    
                        p.append(d_hpx[benchmark][m][c][b][th])
                    plt.figure(i)
                    plt.plot(mat_sizes, p,label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads', marker='o')
                
                    plt.xlabel("# matrix size")           
                    plt.ylabel('MFlops')
                    plt.xscale('log')
                    plt.grid(True, 'both')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    plt.title(benchmark)
                    plt.xticks(np.concatenate((np.arange(2,11)*100, np.arange(2,8)*1000)))
            i=i+1
            plt.savefig(pp, format='pdf',bbox_inches='tight')
            print('')
plt.show()
pp.close() 
        
b='dmatdmatadd'
th=16
filename=hpx_dir+'/cache_miss_results_all.txt'
f=open(filename, 'r')                
r=f.readlines()
for line in r:
    if 'matrix size: ' in line:
        mat_size=int(line.split(': ')[-1].split(',')[0])
    if 'chunk size: ' in line:
        chunk_size=int(line.split(': ')[-1])
    if 'block size row: ' in line:
        b_row=int(line.split(': ')[-1])
    if 'block size col: ' in line:
        b_col=int(line.split(': ')[-1])
    if 'cache-misses:u' in line:
        cm=int(line.split('cache-misses:u')[0].strip().replace(',',''))
    if 'dmatdmatadd benchmark for hpx finished for' in line:
        d_cache[b][mat_size][chunk_size][str(b_row)+'-'+str(b_col)][th]=cm
        
        


#for th in thr:
#    for c in chunk_sizes:
#        for a in [1, 4, 8]:
#            for b in block_sizes:
#                if b.startswith(str(a)):
#                    p=[]
#                    for m in mat_sizes:                    
#                        p.append(d_cache[benchmark][m][c][b][th])
#                    plt.figure(i)
#                    plt.plot(mat_sizes, p,label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads', marker='o')
#                
#                    plt.xlabel("# matrix size")           
#                    plt.ylabel('#misses')
#                    plt.xscale('log')
#                    plt.yscale('log')
#                    plt.grid(True, 'both')
#                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    plt.title(benchmark)
#        i=i+1
#        plt.savefig(pp, format='pdf',bbox_inches='tight')
#        print('')
#plt.show()
#pp.close()         