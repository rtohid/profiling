#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:35:07 2019

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")


hpx_dir='/home/shahrzad/repos/Blazemark/results/counters/'

perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/dmatdmatadd/hpx/06-13-2019'

from matplotlib.backends.backend_pdf import PdfPages


def create_dict_relative(directory):
    thr=[]
    repeats=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmark=''
    benchmarks=[]
    chunk_sizes=[]
    block_sizes={}
    mat_sizes={}
    
    for filename in data_files:
        if '912' in filename:
            (repeat, benchmark, th, runtime, chunk_size, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
            mat_size=mat_size.split(',')[0]
            if benchmark not in benchmarks:
                    benchmarks.append(benchmark)   
                    mat_sizes[benchmark]=[]
                    block_sizes[benchmark]=[]
            if int(mat_size) not in mat_sizes[benchmark]:
                mat_sizes[benchmark].append(int(mat_size))
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat)) 
            if block_size_row+'-'+block_size_col not in block_sizes[benchmark]:
                block_sizes[benchmark].append(block_size_row+'-'+block_size_col)
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
                  
    thr.sort()
    repeats.sort()      
    chunk_sizes.sort()
    benchmarks.sort()   
    
    d_all={}   
    d={}
    for benchmark in benchmarks:  
        mat_sizes[benchmark].sort()
        block_sizes[benchmark].sort()
        d_all[benchmark]={}
        d[benchmark]={}
        for th in thr:
            d_all[benchmark][th]={}
            d[benchmark][th]={}            
            d_all[benchmark][th]={}        
            for bs in block_sizes[benchmark]:
                d_all[benchmark][th][bs]={}
                d[benchmark][th][bs]={}
                for cs in chunk_sizes:
                    d_all[benchmark][th][bs][cs]={}
                    d[benchmark][th][bs][cs]={}
                    d[benchmark][th][bs][cs]['size']=mat_sizes[benchmark]
                    d[benchmark][th][bs][cs]['mflops']=[0]*len(mat_sizes[benchmark])
                    d_all[benchmark][th][bs][cs]['size']=mat_sizes[benchmark]
                    d_all[benchmark][th][bs][cs]['mflops']=[0]*len(mat_sizes[benchmark])
                    d_all[benchmark][th][bs][cs]['counters']=[[]]*len(mat_sizes[benchmark])
                    d[benchmark][th][bs][cs]['counters']=[{'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th}]*len(mat_sizes[benchmark])

                                        
    data_files.sort()    
       
    for filename in data_files:   
        if '912' in filename:
            benchmark=filename.split('/')[-1].split('-')[1]
            th=int(filename.split('/')[-1].split('-')[2])       
            repeat=int(filename.split('/')[-1].split('-')[0])  
            chunk_size=int(filename.split('/')[-1].split('-')[4])         
            block_size=filename.split('/')[-1].split('-')[5]+'-'+filename.split('/')[-1].split('-')[6]
            mat_size=int(filename.split('/')[-1].split('-')[7].replace('.dat','')) 
            s=mat_sizes[benchmark].index(int(mat_size))
            d_all[benchmark][th][block_size][chunk_size]['size'][s]=mat_size
            d[benchmark][th][block_size][chunk_size]['size'][s]=mat_size
    
            f=open(filename, 'r')
            started=False
            results=f.readlines()
            counters={'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th}
            for r in results:        
                print(r)
                if 'idle-rate' in r and 'pool' in r:
                    idle_rate=float(r.strip().split(',')[-2])/100
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters['idle_rate'][th_num]=idle_rate
                    started=True
                elif 'cumulative-overhead' in r and 'pool' in r:
                    cumulative_overhead=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters['cumulative_overhead_time'][th_num]=cumulative_overhead
                elif 'average-overhead' in r and 'pool' in r:
                    average_overhead=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters['average_overhead_time'][th_num]=average_overhead     
                elif 'average,' in r and 'pool' in r:
                    average_time=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters['average_time'][th_num]=average_time
                elif 'cumulative,' in r and 'pool' in r:
                    cumulative=float(r.strip().split(',')[-1])
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters['cumulative_count'][th_num]=cumulative
                elif r.startswith('Done'):
                    if started:
                        d_all[benchmark][th][block_size][chunk_size]['counters'][s].append(counters)
        #                num=len(d_all[benchmark][th][block_size][chunk_size]['counters'])
                        for t in range(th):
                            d[benchmark][th][block_size][chunk_size]['counters'][s]['idle_rate'][t]+=counters['idle_rate'][t]
                            d[benchmark][th][block_size][chunk_size]['counters'][s]['cumulative_overhead_time'][t]+=counters['cumulative_overhead_time'][t]
                            d[benchmark][th][block_size][chunk_size]['counters'][s]['average_overhead_time'][t]+=counters['average_overhead_time'][t]
                            d[benchmark][th][block_size][chunk_size]['counters'][s]['average_time'][t]+=counters['average_time'][t]
                            d[benchmark][th][block_size][chunk_size]['counters'][s]['cumulative_count'][t]+=counters['cumulative_count'][t]               
                        counters={'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th}
                elif r.lstrip().startswith(str(mat_size)):
                    d_all[benchmark][th][block_size][chunk_size]['mflops'][s]=float(r.strip().split(' ')[-1])
                    d[benchmark][th][block_size][chunk_size]['mflops'][s]=float(r.strip().split(' ')[-1])

    return (d, d_all, chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)  
###########################################################################
    
  
           
(d_hpx, d_hpx_all, chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)=create_dict_relative(hpx_dir)                 


i=1
for benchmark in benchmarks:  
    for th in thr:
#        pp = PdfPages(perf_directory+'/hpx_performance_counters_'+str(th)+'.pdf')
        for block_size in block_sizes[benchmark]:
            for chunk_size in chunk_sizes:
                for m in mat_sizes[benchmark]:
                    s=mat_sizes[benchmark].index(m)
                    plt.figure(i)
                    plt.bar(np.arange(1,th+1).tolist(), d_[benchmark][th][block_size][chunk_size]['counters'][s]['idle_rate'], label='idle_rate')
                    plt.xlabel("#threads")      
                    plt.xticks(np.arange(1,th+1).tolist())
                    plt.ylabel('%')
                    plt.title('idle_rate matrix_size:'+str(m))
    #                plt.savefig(pp, format='pdf',bbox_inches='tight')
                    print('')
                    i=i+1
                    plt.figure(i)
                    plt.bar(np.arange(1,th+1).tolist(), d_hpx[benchmark][th][block_size][chunk_size]['counters'][s]['average_time'], label='average_time')
                    plt.xlabel("#threads")     
                    plt.xticks(np.arange(1,th+1).tolist())
                    plt.ylabel('Microseconds')
                    plt.title('average_time matrix_size:'+str(m))
    #                plt.savefig(pp, format='pdf',bbox_inches='tight')
                    print('')
                    i=i+1
                    plt.figure(i)
                    plt.bar(np.arange(1,th+1).tolist(), d_hpx[benchmark][th][block_size][chunk_size]['counters'][s]['cumulative_overhead_time'], label='cumulative_overhead_time')
                    plt.xlabel("#threads") 
                    plt.xticks(np.arange(1,th+1).tolist())                   
                    plt.ylabel('Microseconds')
                    plt.title('cumulative_overhead_time matrix_size:'+str(m))
    #                plt.savefig(pp, format='pdf',bbox_inches='tight')
                    print('')
                    i=i+1
                    plt.figure(i)
                    plt.bar(np.arange(1,th+1).tolist(), d_hpx[benchmark][th][block_size][chunk_size]['counters'][s]['cumulative_count'], label='cumulative_count')
                    plt.xlabel("#threads") 
                    plt.xticks(np.arange(1,th+1).tolist())                   
            #        plt.ylabel('')
                    plt.title('cumulative_count matrix_size:'+str(m))
    #                plt.savefig(pp, format='pdf',bbox_inches='tight')
                    print('')
                    i=i+1
                    plt.figure(i)
                    plt.bar(np.arange(1,th+1).tolist(), d_hpx[benchmark][th][block_size][chunk_size]['counters'][s]['average_overhead_time'], label='average_overhead_time')
                    plt.xlabel("#threads")    
                    plt.xticks(np.arange(1,th+1).tolist())                   
                    plt.ylabel('Microseconds')
                    plt.title('average_overhead_time matrix_size:'+str(m))
                    i=i+1


i=1
for benchmark in benchmarks:  
    for th in thr:
#        pp = PdfPages(perf_directory+'/hpx_performance_counters_'+str(th)+'.pdf')
        for block_size in block_sizes[benchmark]:
            for chunk_size in chunk_sizes:
                for m in mat_sizes[benchmark]:
                    for t in range(1,20):
                        s=mat_sizes[benchmark].index(m)
                        plt.figure(i)
                        print(sum(d_hpx_all[benchmark][th][block_size][chunk_size]['counters'][s][t]['cumulative_count']))
                        plt.bar(np.arange(1,th+1).tolist(), d_hpx_all[benchmark][th][block_size][chunk_size]['counters'][s][t]['cumulative_count'], label='cumulative_count')
                        plt.xlabel("#threads") 
                        plt.xticks(np.arange(1,th+1).tolist())                   
                #        plt.ylabel('')
                        plt.title('cumulative_count matrix_size:'+str(m))
                        i=i+1
                        plt.figure(i)

                        plt.bar(np.arange(1,th+1).tolist(), d_hpx_all[benchmark][th][block_size][chunk_size]['counters'][s][t]['idle_rate'], label='idle_rate')
                        plt.xlabel("#threads")      
                        plt.xticks(np.arange(1,th+1).tolist())
                        plt.ylabel('%')
                        plt.title('idle_rate matrix_size:'+str(m))
        #                plt.savefig(pp, format='pdf',bbox_inches='tight')
                        print('')
                        i=i+1
                        plt.figure(i)
                        plt.bar(np.arange(1,th+1).tolist(), d_hpx[benchmark][th][block_size][chunk_size]['counters'][s][t]['average_time'], label='average_time')
                        plt.xlabel("#threads")     
                        plt.xticks(np.arange(1,th+1).tolist())
                        plt.ylabel('Microseconds')
                        plt.title('average_time matrix_size:'+str(m))
        #                plt.savefig(pp, format='pdf',bbox_inches='tight')
                        print('')
                        i=i+1
                        plt.figure(i)
                        plt.bar(np.arange(1,th+1).tolist(), d_hpx[benchmark][th][block_size][chunk_size]['counters'][s][t]['cumulative_overhead_time'], label='cumulative_overhead_time')
                        plt.xlabel("#threads") 
                        plt.xticks(np.arange(1,th+1).tolist())                   
                        plt.ylabel('Microseconds')
                        plt.title('cumulative_overhead_time matrix_size:'+str(m))
        #                plt.savefig(pp, format='pdf',bbox_inches='tight')
                        print('')
                        i=i+1
                        