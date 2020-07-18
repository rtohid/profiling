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

sizes=[1000000]    
directory='/home/shahrzad/Spyder/Blazemark/results/data/'+str(sizes[0])+'/6'
data_files=glob.glob(directory+'/*.dat')
thr=np.arange(16).tolist()
#    sizes=[100,1000,10000,100000,1000000,10000000]

d_all={}
for s in sizes:
    d_all[s]={}

multiplyers=[]
chunk_sizes=[]    
for filename in data_files:        
    if filename.count('-')==2:
        (benchmark, cores, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
#            print(benchmark, cores, runtime)
        k=runtime
    else:
        (benchmark, cores, runtime, chunk_size, multiplyer, threshold) = filename.split('/')[-1].replace('.dat','').split('-')
#            print(benchmark, cores, runtime, chunk_size, multiplyer, threshold)
        
        k=chunk_size+'-'+multiplyer+'-'+threshold
        
        if multiplyer not in multiplyers:
            multiplyers.append(multiplyer)
            
        if chunk_size not in chunk_sizes:
            chunk_sizes.append(chunk_size)
            
        f=open(filename, 'r')
        results=f.readlines()[3:]
        s=results[0].strip().split(' ')[0]
#            print("s:",s)
        if k not in d_all[int(s)].keys():            
            d_all[int(s)][k]={'idle_rate':[[0]*(t+1) for t in thr], 'average_time':[[0]*(t+1) for t in thr],
                  'cumulative_count':[[0]*(t+1) for t in thr], 'cumulative_overhead':[[0]*(t+1) for t in thr],
                  'average_overhead':[[0]*(t+1) for t in thr]}
        for r in results:
            if 'idle-rate' in r and 'pool' in r:
                idle_rate=float(r.strip().split(',')[-2])/100
                th_num=int(r.strip().split('thread#')[1].split('}')[0])
#                    print(idle_rate)
                d_all[int(s)][k]['idle_rate'][int(cores)-1][th_num]=idle_rate
            if 'average,' in r and 'pool' in r:
                average_time=float(r.strip().split(',')[-2])/1000
                th_num=int(r.strip().split('thread#')[1].split('}')[0])
#                    print(idle_rate)
                d_all[int(s)][k]['average_time'][int(cores)-1][th_num]=average_time
            if 'cumulative,' in r and 'pool' in r:
                cumulative=float(r.strip().split(',')[-1])
                th_num=int(r.strip().split('thread#')[1].split('}')[0])
#                    print(idle_rate)
                d_all[int(s)][k]['cumulative_count'][int(cores)-1][th_num]=cumulative
            if 'cumulative-overhead' in r and 'pool' in r:
                cumulative_overhead=float(r.strip().split(',')[-2])/1000
                th_num=int(r.strip().split('thread#')[1].split('}')[0])
#                    print(idle_rate)
                d_all[int(s)][k]['cumulative_overhead'][int(cores)-1][th_num]=cumulative_overhead
            if 'average-overhead' in r and 'pool' in r:
                average_overhead=float(r.strip().split(',')[-2])/1000
                th_num=int(r.strip().split('thread#')[1].split('}')[0])
#                    print(idle_rate)
                d_all[int(s)][k]['average_overhead'][int(cores)-1][th_num]=average_overhead
    
#chunk_sizes=[1, 8, 16, 32, 64, 128, 256]
#chunk_sizes=[1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 32, 64, 128, 256]
threshold='1000'
#for m in [1,2,4,8,10]:
m=1
for p_counter in ['idle_rate', 'average_time', 'cumulative_count', 'cumulative_overhead', 'average_overhead']:
    pp = PdfPages(directory+'/blazemark_hpx_config_'+p_counter+'.pdf')

    i=1

    for s in sizes:        
        for c in chunk_sizes:
            k=str(c)+'-'+str(m)+'-'+threshold
            if '-'+str(m)+'-1000' in k:
                plt.figure(i)

                for t in thr: 
                    ax=plt.subplot(4,4,t+1)

                    if t%4 !=0:
                        if t/4>3:
                            ax.tick_params(axis='both', left='off',right='off',labelleft='off', labeltop='off',labelright='off')

                        else:
                            ax.tick_params(axis='both', bottom='off', left='off',right='off',labelleft='off', labeltop='off',labelright='off', labelbottom='off')
                    else:
                        if t==12:
                            plt.xlabel('#number of cores')
                            if p_counter=='average_time':
                                plt.ylabel(p_counter+' (microseconds)')
                            if p_counter=='idle_rate':
                                plt.ylabel(p_counter+' (%)')
                            if p_counter=='cumulative_count':
                                plt.ylabel(p_counter)
                            if p_counter=='cumulative_overhead':
                                plt.ylabel(p_counter+' (microseconds)')
                            if p_counter=='average_overhead':
                                plt.ylabel(p_counter+' (microseconds)')
                                
                        else:
                            ax.tick_params(axis='both', bottom='off', labelbottom='off')
                    plt.xlim([1,16])

                    if p_counter=='idle_rate':
                        plt.ylim([0,100])

                    plt.bar(thr[0:t+1], d_all[s][k][p_counter][t],label="chunk size= "+str(k.split('-')[0])+" multiplyer="+str(k.split('-')[1]))   
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                    
                fig=plt.gcf()
                fig.suptitle(str(i)+':        '+date_str+'    vector size: '+str(s)+'      threshold=1000')
                i=i+1
        
                plt.savefig(pp, format='pdf',bbox_inches='tight')
                print('')

    plt.show()
    pp.close()
    
    
    
    