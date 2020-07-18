#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:21:00 2020

@author: shahrzad
"""

#!/usr/bin/env python3
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


def create_dict_stencil(directories,to_csv=False):
    thr=[]
    data_files=[]
    
    for directory in directories:
        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]

    nps=[]
    nxs=[]
    nodes=[]
    gps=[]
    if to_csv:
        f_csv=open('/home/shahrzad/repos/Blazemark/data/stencil_data_perf_all.csv','w')
        f_writer=csv.writer(f_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(['node','grid_points','num_threads','num_partitions','partition_size','grain_size','work_per_core','num_tasks','execution_time'])

    for filename in data_files:       
        (node, _, th, npo, nx) = filename.split('/')[-1].replace('.dat','').split('_')                 
        th=int(th)
        npo=int(npo)
        nx=int(nx)
       
        if node not in nodes:
            nodes.append(node)
        if npo not in nps:
            nps.append(npo)              
        if nx not in nxs:
            nxs.append(nx)
        if th not in thr:
            thr.append(th)
        if nx*npo not in gps:
            gps.append(nx*npo)
            
    nodes.sort()
    nxs.sort()
    nps.sort()
    gps.sort()
    thr.sort()              
                                                           
    data_files.sort()   
    
    for filename in data_files:   
        f=open(filename, 'r')
                 
        result=f.readlines()
        if len(result)==2:
            (node, _, th, npo, nx) = filename.split('/')[-1].replace('.dat','').split('_')                 
            th=float(th)
            npo=float(npo)
            nx=float(nx)
           
            r=result[1].split(',')
            r=[rr.strip() for rr in r]
            execution_time=float(r[1])*1e6
            num_tasks=npo
            grain_size=nx        
            w_c=math.ceil(num_tasks/th)*grain_size
            f_writer.writerow([node,npo*nx,th,npo,nx,grain_size,w_c,num_tasks,execution_time])

    if to_csv:
        f_csv.close()
#    return (data, d, thr, iter_lengths, num_iterations)  

marvin_dir='/home/shahrzad/repos/Blazemark/data/stencil/marvin'
#medusa_dir='/home/shahrzad/repos/Blazemark/data/grain_size/medusa'
create_dict_stencil([marvin_dir],1)


def my_func_g_3(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,1]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/(ps*N))*((n_t-1))*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)


titles=['node','grid_points','num_threads','num_partitions','partition_size','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/stencil_data_perf_all.csv'


dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()

grid_points=dataframe['grid_points'].drop_duplicates().values
grid_points.sort()

            
for node in nodes:
    node_selected=dataframe['node']==node
    df_n_selected=dataframe[node_selected][titles[1:]]
    
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()

    grid_points=dataframe['grid_points'].drop_duplicates().values
    grid_points.sort()

    array=df_n_selected.values
    array=array.astype(float)
    
    data_size=int(np.shape(array)[0])
    print(data_size)
    
    train_size=int(1*np.ceil(data_size*0.6))
    tes_size=data_size-train_size
    
    per = np.random.permutation(data_size)
    train_set=array[per[0:train_size],:-1] 
    train_labels=array[per[0:train_size],-1]  
    test_set=array[per[train_size:],:-1]  
    test_labels=array[per[train_size:],-1]  
    
    param_bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,np.inf])
#        param_bounds=([0,-np.inf],[np.inf,np.inf])

    popt_3, pcov=curve_fit(my_func_g_3,train_set,train_labels,method='trf',bounds=param_bounds)

    i=1

    for ps in grid_points:
        
        array_ps=train_set[train_set[:,0]==ps]
        labels_ps=train_labels[train_set[:,0]==ps]
        
        for th in thr:
            new_array=array_ps[array_ps[:,1]==th]
            new_labels=labels_ps[array_ps[:,1]==th]
            if np.shape(new_array)[0]>3:
                plt.figure(i)
#               
        
                plt.scatter(new_array[:,4],new_labels,marker='.',label='true')
#               
        
                plt.xlabel('grain size')
                plt.ylabel('execution time')
                plt.xscale('log')
                plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
                plt.axvline(ps/th,color='gray',linestyle='dotted')  
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
                i=i+1        
        
    i=1

    for ps in grid_points[-4:]:
        
        array_ps=test_set[test_set[:,0]==ps]
        labels_ps=test_labels[test_set[:,0]==ps]
        
        for th in thr:
            new_array=array_ps[array_ps[:,1]==th]
            new_labels=labels_ps[array_ps[:,1]==th]
            if np.shape(new_array)[0]>3:
                plt.figure(i)
#                z_1=my_func_g_1(new_array,*popt_1)
#                z_2=my_func_g_2(new_array,*popt_2)
#                z_3=my_func_g_3(new_array,*popt_3)
#                z_4=my_func_g_4(new_array,*popt_4)
        
                plt.scatter(new_array[:,4],new_labels,marker='.',label='true')
#                plt.scatter(new_array[:,5],z_1,marker='.',label='pred1')
#                plt.scatter(new_array[:,5],z_2,marker='.',label='pred2')
#                plt.scatter(new_array[:,4],z_3,marker='.',label='pred3')
#                plt.scatter(new_array[:,5],z_4,marker='.',label='pred4')
        
                plt.xlabel('grain size')
                plt.ylabel('execution time')
                plt.xscale('log')
                plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
                plt.axvline(ps/th,color='gray',linestyle='dotted')  
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
                i=i+1        