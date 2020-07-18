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


def create_dict(directories,to_csv=True,data_filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'):
    thr=[]
    data_files=[]
    
    for directory in directories:
        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]
    
    chunk_sizes=[]
    num_iterations=[]
    iter_lengths=[]
    nodes=[]
    
    if to_csv:
        f_csv=open(data_filename,'w')
        f_writer=csv.writer(f_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time'])

    for filenames in data_files:
        filename=filenames.replace('marvin_old','c7')
        filename=filename.replace('_idle','')

        if 'seq' in filename:
            (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=0
            th=1         
        elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
            (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=1
        else:
            (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
        if 'spt' in filename:
            chunk_size=-1
            
        chunk_size=int(chunk_size)
        th=int(th)
        iter_length=int(iter_length)
        num_iteration=int(num_iteration)
        if node not in nodes:
            nodes.append(node)
        if iter_length not in iter_lengths:
            iter_lengths.append(iter_length)
        if num_iteration not in num_iterations:
            num_iterations.append(num_iteration)        
        if th not in thr:
            thr.append(th)
        if chunk_size not in chunk_sizes:
            chunk_sizes.append(chunk_size)

    nodes.sort()
    num_iterations.sort()
    iter_lengths.sort()
    thr.sort()                  
                                                           
    data_files.sort()   
    problem_sizes=[]
#    marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
#    excludes=[marvin_dir+'/marvin_grain_size_1_1_2000_50000.dat',marvin_dir+'/marvin_grain_size_1_1_1000_100000.dat']
    excludes=[] 
    for filenames in data_files:   
        if filenames not in excludes:             
            f=open(filenames, 'r')
                     
            result=f.readlines()
            filename=filenames.replace('marvin_old','c7')
            filename=filename.replace('_idle','')

            if len(result)!=0:
                avg=0
                if 'seq' in filename:
                    (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                    chunk_size=0
                    th=1   
                elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
                    (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                    chunk_size=1
                else:
                    (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                if 'spt' in filename:
                    chunk_size=-1
                chunk_size=float(chunk_size)        
                th=float(th)       
                iter_length=float(iter_length)
                num_iteration=float(num_iteration)    
                first=True
                for r in [r for r in result if r!='\n' and r!='split type:idle\n']:  
                    if not first:
                        avg+=float(r.split('in ')[1].split('microseconds')[0].strip())
                    else:
                        first=False
                avg=avg/(len([r for r in result if r!='\n'])-1)
        #        d[node][num_iteration][th][chunk_size][iter_length]=avg/5
                problem_size=num_iteration*(iter_length)
                if problem_size not in problem_sizes:
                    problem_sizes.append(problem_size)
                grain_size=chunk_size*(iter_length)
                if chunk_size!=0:
                    num_tasks=np.ceil(num_iteration/chunk_size)
                    L=np.ceil(num_tasks/th)
                    w_c=L*grain_size
                    if th==1:
                        w_c=num_iteration*(iter_length)
                    if num_tasks%th==1 and num_iteration%chunk_size!=0:
    #                    w_c_1=problem_size+(1-th)*(L-1)*grain_size
                        w_c=(L-1)*grain_size+(num_iteration%chunk_size)*(iter_length)
                else:
                    num_tasks=0
                    L=0
                    w_c=0
                
                
                f_writer.writerow([node,problem_size,num_iteration,th,chunk_size,iter_length,grain_size,w_c,num_tasks,avg])

    if to_csv:
        f_csv.close()
#    return (data, d, thr, iter_lengths, num_iterations)  

marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
medusa_dir='/home/shahrzad/repos/Blazemark/data/grain_size/medusa'
tameshk_dir='/home/shahrzad/repos/Blazemark/data/grain_size/tameshk'
marvin_old_dir_all='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/all_cores/spt_min_0'
marvin_old_dir_idle='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/idle_cores/2/'

#results_dir='/home/shahrzad/repos/Blazemark/results/grain_size'
#create_dict([results_dir],1)

create_dict([marvin_dir,medusa_dir,tameshk_dir])
create_dict([marvin_old_dir_all],data_filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all_cores.csv')
create_dict([marvin_old_dir_idle],data_filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_idle_cores.csv')


#grain size data split by all cores
titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename_all='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all_cores.csv'
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

dataframe = pandas.read_csv(filename_all, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()

#splittable task results
node='c7-spt'
node_selected=dataframe['node']==node
iter_selected=dataframe['iter_length']==1
th_selected=dataframe['num_threads']>=1
cs_selected=dataframe['chunk_size']==-1

df_n_selected=dataframe[node_selected & cs_selected & iter_selected & th_selected][titles[1:]]

threads={}
thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()
threads[node]=thr

problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
problem_sizes.sort()

spt_results_all={}
spt_results_all[node]={}

array=df_n_selected.values
for ps in problem_sizes:
    array_ps=array[array[:,0]==ps]
    spt_results_all[node][ps]={}
    for th in thr:
        array_t=array_ps[array_ps[:,2]==th]
        spt_results_all[node][ps][th]=array_t[:,-1]


##grain size data split by idle cores
titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename_idle='/home/shahrzad/repos/Blazemark/data/grain_data_perf_idle_cores.csv'
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

dataframe = pandas.read_csv(filename_idle, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()

#splittable task results
node='c7-spt'
node_selected=dataframe['node']==node
iter_selected=dataframe['iter_length']==1
th_selected=dataframe['num_threads']>=1
cs_selected=dataframe['chunk_size']==-1

df_n_selected=dataframe[node_selected & cs_selected & iter_selected & th_selected][titles[1:]]

threads={}
thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()
threads[node]=thr

problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
problem_sizes.sort()

spt_results_idle={}
spt_results_idle[node]={}

array=df_n_selected.values
for ps in problem_sizes:
    array_ps=array[array[:,0]==ps]
    spt_results_idle[node][ps]={}
    for th in thr:
        array_t=array_ps[array_ps[:,2]==th]
        spt_results_idle[node][ps][th]=array_t[:,-1]
        

node='marvin'

titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)
    
node_selected=dataframe['node']==node
nt_selected=dataframe['num_tasks']>=1
iter_selected=dataframe['iter_length']==1
th_selected=dataframe['num_threads']>=1
df_n_selected=dataframe[node_selected & nt_selected & iter_selected & th_selected][titles[1:]]

thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()
threads[node]=thr

i=1
array=df_n_selected.values
for ps in problem_sizes:
    array_ps=array[array[:,0]==ps]
    if np.shape(array_ps)[0]!=0:
        for th in thr:
            plt.figure(i)
            array_t=array_ps[array_ps[:,2]==th]
            plt.scatter(array_t[:,5],array_t[:,-1])
            plt.axhline(spt_results_all['c7-spt'][ps][th], color='green', label='all cores')
            plt.axhline(spt_results_idle['c7-spt'][ps][th], color='red', label='idle cores')

            plt.axvline(ps/th,color='gray',linestyle='dashed')
            plt.xlabel('Grain size')
            plt.ylabel('Execution time')
            plt.xscale('log')
            plt.title('ps='+str(int(ps))+' '+str(int(th))+' threads')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            i=i+1
            plt.savefig(perf_dir+'/splittable/all_idle_cores/2/'+node+'_'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')
