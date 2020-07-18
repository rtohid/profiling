c#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:19:59 2020

@author: shahrzad
"""

import pandas
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import glob
import matplotlib.tri as mtri
from mpl_toolkits import mplot3d
import math
from scipy.optimize import curve_fit
from collections import Counter
from scipy.optimize import nnls
import csv
from scipy.optimize import nnls
from sklearn.metrics import r2_score


perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'
def grain_dict(array,avg=False):
    g_dict={}
    
    g=array[:,3]
    p=array[:,-1]
    t=array[:,2]
    nt=array[:,-2]
    
    for i in range(len(g)):
        if g[i] not in g_dict.keys():
            g_dict[g[i]]={}
        if t[i] not in g_dict[g[i]].keys():
            g_dict[g[i]][t[i]]=[[],[]]
        g_dict[g[i]][t[i]][0].append(p[i])
        g_dict[g[i]][t[i]][1].append(nt[i])

    if avg:
        for gd in g_dict.keys():
            for td in g_dict[gd].keys():
                g_dict[gd][td][0]=sum(g_dict[gd][td][0])/len(g_dict[gd][td][0])
                g_dict[gd][td][1]=np.ceil(sum(g_dict[gd][td][1])/len(g_dict[gd][td][1]))
    return g_dict

def create_dict_reference(directory):
    thr=[]
    nodes=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmark=''
    benchmarks=[]
    
    for filename in data_files:
        try:
            (node, benchmark, th) = filename.split('/')[-1].replace('.dat','').split('-')   
        except:
            (node, benchmark, th, ref) = filename.split('/')[-1].replace('.dat','').split('-')  
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)                
        if int(th) not in thr:
            thr.append(int(th))
        if node not in nodes:
            nodes.append(node)        
                  
    thr.sort()
    benchmarks.sort()      
    nodes.sort()      
    
    d_all={}   
    for node in nodes:
        d_all[node]={}
        for benchmark in benchmarks:  
            d_all[node][benchmark]={}
            for th in thr:
                d_all[node][benchmark][th]={}     
                                            
    data_files.sort()        
    for filename in data_files:                
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        try:
            (node, benchmark, th) = filename.split('/')[-1].replace('.dat','').split('-')   
        except:
            (node, benchmark, th, ref) = filename.split('/')[-1].replace('.dat','').split('-')          
        th = int(th)
        size=[]
        mflops=[]    
        for r in result:        
            if "N=" in r or '/' in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
            
        d_all[node][benchmark][th]['size']=size
        d_all[node][benchmark][th]['mflops']=mflops
 
            
    return d_all                           

hpx_dir_ref_0='/home/shahrzad/repos/Blazemark/data/matrix/09-15-2019/reference-chunk_size_fixed/'         
hpx_dir_ref_0='/home/shahrzad/repos/Blazemark/data/matrix/c7/reference/4-10-2020-wo-numa'
hpx_dir_ref='/home/shahrzad/repos/Blazemark/data/matrix/c7/reference/'


d_hpx_ref=create_dict_reference(hpx_dir_ref)  
#d_hpx_ref_0=create_dict_reference(hpx_dir_ref_0)  
  
b_filename='/home/shahrzad/repos/Blazemark/data/data_perf_all.csv'
titles=['runtime','node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','work_per_core','w1','w2','w3','w4','w5','w6','w7','w8','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops','include']

dataframe = pandas.read_csv(b_filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[3:]:
    dataframe[col] = dataframe[col].astype(float)



runtime='hpx'
benchmark='dmatdmatadd'
spt_node='marvin_old_spt'  
threads={}
spt_results_all={}
spt_results_all[spt_node]={}
b='4-256'
spt_results_all[spt_node][b]={}
spt_results_all[spt_node][b][benchmark]={}

threads[spt_node]={}
included=dataframe['include']==1
node_selected=dataframe['node']==spt_node
df_n_selected=dataframe[node_selected & included]
benchmark_selected=dataframe['benchmark']==benchmark
rt_selected=dataframe['runtime']==runtime
num_threads_selected=dataframe['num_threads']<=8
df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected & rt_selected]         
block_selected_r=df_nb_selected['block_size_row']==4
block_selected_c=df_nb_selected['block_size_col']!=64
df_nb_selected=df_nb_selected[ block_selected_r | block_selected_c]
df_nb_selected=df_nb_selected[ block_selected_r ]#| block_selected_c]

matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
matrix_sizes.sort()
thr=df_nb_selected['num_threads'].drop_duplicates().values
thr.sort()
threads[spt_node][benchmark]=thr

features=['chunk_size','num_blocks','num_threads','grain_size','block_size_row','block_size_col','work_per_core','num_tasks','execution_time']


for m in matrix_sizes:
    m_selected=df_nb_selected['matrix_size']==m
    spt_results_all[spt_node][b][benchmark][m]={}
    df_selected=df_nb_selected[m_selected][features]
    array_b=df_selected.values

    for th in thr:
        spt_results_all[spt_node][b][benchmark][m][th]=array_b[array_b[:,2]==th][:,-1]


runtime='hpx'
benchmark='dmatdmatadd'
spt_node_idle='marvin_old_spt_idle'  
threads={}
spt_results_idle={}
spt_results_idle[spt_node_idle]={}
b='4-256'
spt_results_idle[spt_node_idle][b]={}
spt_results_idle[spt_node_idle][b][benchmark]={}

threads[spt_node]={}
included=dataframe['include']==1
node_selected=dataframe['node']==spt_node
df_n_selected=dataframe[node_selected & included]
benchmark_selected=dataframe['benchmark']==benchmark
rt_selected=dataframe['runtime']==runtime
num_threads_selected=dataframe['num_threads']<=8
df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected & rt_selected]         
block_selected_r=df_nb_selected['block_size_row']==4
block_selected_c=df_nb_selected['block_size_col']!=64
df_nb_selected=df_nb_selected[ block_selected_r | block_selected_c]
df_nb_selected=df_nb_selected[ block_selected_r ]#| block_selected_c]

matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
matrix_sizes.sort()
thr=df_nb_selected['num_threads'].drop_duplicates().values
thr.sort()
threads[spt_node][benchmark]=thr

features=['chunk_size','num_blocks','num_threads','grain_size','block_size_row','block_size_col','work_per_core','num_tasks','execution_time']


for m in matrix_sizes:
    m_selected=df_nb_selected['matrix_size']==m
    spt_results_idle[spt_node_idle][b][benchmark][m]={}
    df_selected=df_nb_selected[m_selected][features]
    array_b=df_selected.values

    for th in thr:
        spt_results_idle[spt_node_idle][b][benchmark][m][th]=array_b[array_b[:,2]==th][:,-1]
 

node='marvin'
node_selected=dataframe['node']==node
df_n_selected=dataframe[node_selected & included]
benchmark_selected=dataframe['benchmark']==benchmark
rt_selected=dataframe['runtime']==runtime
num_threads_selected=dataframe['num_threads']<=8
df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected & rt_selected]         
block_selected_r=df_nb_selected['block_size_row']==4
block_selected_c=df_nb_selected['block_size_col']==256
df_nb_selected=df_nb_selected[ block_selected_r & block_selected_c]

matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
matrix_sizes.sort()

i=1
for m in matrix_sizes:
    m_selected=df_nb_selected['matrix_size']==m
    features=['chunk_size','num_blocks','num_threads','grain_size','block_size_row','block_size_col','work_per_core','num_tasks','execution_time']
    df_selected=df_nb_selected[m_selected][features]

    array_b=df_selected.values
    for th in thr:
        k=d_hpx_ref['marvin_old'][benchmark][th]['size'].index(m)

        plt.figure(i)
        new_array=array_b[array_b[:,2]==th][:,:-1]
        new_labels=array_b[array_b[:,2]==th][:,-1]
        plt.scatter(new_array[:,3],new_labels,color='blue',label='true',marker='.')
        plt.axhline(spt_results_all[spt_node][b][benchmark][m][th],color='green',label='all cores')
        plt.axhline(spt_results_idle[spt_node_idle][b][benchmark][m][th],color='red',label='idle cores')
        plt.axhline((m**2)/d_hpx_ref['marvin_old'][benchmark][th]['mflops'][k],color='purple',label='reference')
        plt.axvline((m**2)/th,color='gray',linestyle='dashed')            
        plt.ylabel('Execution time')       
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('matrix size:'+str(int(m))+' '+str(int(th))+' threads')
        plt.savefig(perf_dir+'/blazemark/splittable/all_idle_cores_ref/2/'+node+'_spt_'+str(int(m))+'_'+str(int(th))+'.png',bbox_inches='tight')
        i=i+1
        
