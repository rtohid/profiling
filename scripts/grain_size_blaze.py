#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:35:23 2020

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

titles_grain=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename_grain='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hxp_for_loop/1/all/'


dataframe_grain = pandas.read_csv(filename_grain, header=0,index_col=False,dtype=str,names=titles_grain)
for col in titles_grain[1:]:
    dataframe_grain[col] = dataframe_grain[col].astype(float)

nodes=dataframe_grain['node'].drop_duplicates().values
nodes.sort()

problem_sizes=dataframe_grain['problem_size'].drop_duplicates().values
problem_sizes.sort()


def find_flat(x,y):
    y_prev=y[0]
    prev_i=0
    regions=[]
    for i in range(1,np.shape(x)[0]):
        if abs(y[i]-y_prev)>0.04*y_prev:
            if i-prev_i>5:
                regions.append([x[prev_i],x[i]])
            prev_i=i

        y_prev=y[i]
    return regions
        

i=1
def my_func_g_3(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/(ps*N))*((n_t-1))*np.heaviside(N-n_t,1)

for node in nodes:
    np.random.seed(0)                

    node_selected_grain=dataframe_grain['node']==node
    df_n_selected_grain=dataframe_grain[node_selected_grain][titles_grain[1:]]
    
    thr=df_n_selected_grain['num_threads'].drop_duplicates().values
    thr.sort()

    problem_sizes=df_n_selected_grain['problem_size'].drop_duplicates().values
    problem_sizes.sort()

    array_grain=df_n_selected_grain.values
    array_grain=array_grain.astype(float)
        

    for th in thr:
        all_regions={}

        new_array_grain=array_grain[array_grain[:,2]==th]
        new_labels_grain=array_grain[array_grain[:,2]==th]
            
        for ps in problem_sizes[-20:]:    
            
            array_ps_grain=new_array_grain[new_array_grain[:,0]==ps][:,:-1]
            labels_ps_grain=new_labels_grain[new_labels_grain[:,0]==ps][:,-1]
            
            a_s=np.argsort(array_ps_grain[:,5])
            for ir in range(np.shape(array_ps_grain)[1]):
                array_ps_grain[:,ir]=array_ps_grain[a_s,ir]
            labels_ps_grain=labels_ps_grain[a_s]    
            
            n_t=array_ps_grain[:,-1]
            M=np.minimum(n_t,th) 
            L=np.ceil(n_t/th)
            w_c=array_ps_grain[:,-2]
            prs=array_ps_grain[:,0]
#            
#            param_bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,np.inf])
#            popt_3, pcov=curve_fit(my_func_g_3,array_ps_grain,labels_ps_grain,method='trf',bounds=param_bounds)
#            [alpha,gamma,d,h,q]=popt_3
#
#            plt.plot(array_ps_grain[:,5],(th-1)*(th-2)*q/prs,label='q*(n-1)*(n-2)/ps')
#            plt.plot(array_ps_grain[:,5],alpha*L,label='alpha*L')
#            plt.plot(array_ps_grain[:,5],(1+(gamma)*(M-1))*(w_c),label='(1+(gamma)*(M-1))*(w_c)')
#            plt.plot(array_ps_grain[:,5],h*n_t*(th-1)*np.heaviside(n_t-th,1),label='h*n_t*(N-1)*np.heaviside(n_t-N,1)')
#            plt.plot(array_ps_grain[:,5],(d/(ps*th))*((n_t-1))*np.heaviside(th-n_t,1),label='q*(n-1)*(n-2)/ps')
#            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#            plt.plot(array_ps[:,5],alpha*L+(1+(gamma)*(M-1))*(w_c),label='alpha*L')
#            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#            plt.scatter(array_ps[:,5],w_c/ps,label='w_c/ps',marker='.')
#            plt.xscale('log')
#            plt.grid(True,'both')
#
#            
#            plt.scatter(array_ps[:,5],w_c,label='w_c',marker='.')
#            plt.plot(array_ps[:,5],L,label='L')
##            plt.plot(array_ps[:,5],(n_t),label='(n_t)')
#            plt.axvline(np.sqrt(alpha*ps/(0.1*th)),color='purple')
#            plt.axvline((ps/(th*(th+1)))+(0.01/(th+1)),color='green')
#
#            plt.plot(array_ps[:,5],alpha*L,label='alpha*L')
#           
##            plt.plot(array_ps[:,5],alpha*ps/(th*(array_ps[:,5]**2)),label='alpha*ps/(g*th)')
#
#            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            
            if np.shape(np.unique(array_ps_grain[:,5]))[0]>10 and np.shape(array_ps_grain)[0]>10:
#                print(ps)
                plt.figure(i)
                plt.axes([0, 0, 4, 1])
                ll=np.sqrt(0.05*ps*2.85/th)
                lambda_s=0.1
                k_s=(1/lambda_s)+1
                lr=ps*lambda_s/th-1

                plt.scatter(array_ps_grain[:,5], labels_ps_grain/ps,marker='.',label='speed-up')   
#                plt.scatter(array_ps_grain[:,5], (w_c-ps/th)/(ps/th),marker='.',label='imbalance ratio')   
                plt.scatter(array_ps_grain[:,5], (w_c-ps/M)/(ps/M),marker='.',label='imbalance ratio') 
                plt.scatter(array_ps_grain[:,5], (w_c-ps/th)/(ps/th),marker='.',label='imbalance ratio')   

#                plt.scatter(1, 1/(np.arange(1,th)-1),marker='.',label='M=1')  
                
#                g=np.arange(1,ps)
#                k=np.arange(2,np.ceil(ps/th))
#                plt.scatter(k, 1/(k-1),marker='.',label='M='+str(int(t)))   

#                for t in range(2,th+1):    
#                    plt.figure(i)
#
#                    plt.scatter(t, 1/(t-1),marker='.',label='M='+str(int(t)))   
#                plt.scatter(np.arange(2,np.ceil(ps/th)), -0.0001+1/(np.arange(2,ps/th)-1),marker='.',label='k>1',color='blue')   
#                x=np.arange(np.ceil(ps/th),ps+1)
#                plt.scatter(x, (x-ps/th)/(ps/th),marker='.',label='1/speed-up')   

#                plt.scatter(array_ps_grain[:,5], labels_ps_grain/ps,marker='.',label=str(int(th))+' threads')   
#                plt.scatter(array_ps[:,5]*100/ps, my_func_g_3(array_ps,*popt_3),marker='.',label='fit')   
#                plt.axvline((ps/(th*(th+1)))+(0.01/(th+1)),color='green')
#                plt.axvline(np.sqrt(alpha*ps/(0.1*th)),color='purple')
                plt.xlabel('problem_size/grain_size')
                plt.xlabel('Grain size')
#                plt.xlabel(r'$k=\left\lceil{\frac{num\_{tasks}}{N}}\right \rceil$')
                plt.scatter(ps/th,0,marker='.',c='C0')
#                plt.axvline(ll,color='green') 
#                plt.axvline(lr,color='green') 
#                plt.ylabel('1/speedup')
#                plt.ylabel('Execution Time(microseconds)')
                plt.ylabel('Imbalance Ratio')
                plt.xscale('log')

#                plt.title('Problem size: '+str(int(ps)))
                plt.title('Problem size: '+str(int(ps))+'  '+str(int(th))+' threads')
                plt.grid(True,'both')
#                for j in range(np.shape(np.unique(L))[0]):  
#                    k=np.unique(L)[j]
#                    if j>np.shape(np.unique(L))[0]-5 or (j<3 and j!=0):
#                        plt.annotate('k='+str(int(k)), ((1+0.1/j)*ps/(k*th),5),textcoords="offset points", xytext=(20,0), ha='center',rotation=90)  
#                    if k>1:    
#                        plt.axvline(ps/(k*th),color='green',linestyle='dashed')
#                        plt.axvline(ps/((k-1)*th),color='green',linestyle='dashed')
#
#                plt.annotate('k=1', ((1+0.1)*ps/(th),5),textcoords="offset points", xytext=(20,0), ha='center',rotation=90)  
#
###
#                for j in range(1,th):
#                    plt.axvline(np.ceil(ps/j),color='purple',linestyle='dashed')
#                    plt.annotate('M='+str(j+1), (ps/j/1.3,0.5),textcoords="offset points", xytext=(20,0), ha='center',rotation=90) 
#                             
#                    
                plt.axvline(ps/(th*(1+np.ceil(1/0.2))),color='green',linestyle='dotted')  

#                plt.axvline(ps/th,color='gray',linestyle='dotted')  
#                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                plt.savefig(perf_dir+'w_c_all.png',bbox_inches='tight')

#            plt.figure(i)
#
            plt.savefig(perf_dir+str(int(ps))+'_'+str(int(th))+'_all.png',bbox_inches='tight')

                i=i+1
#                all_regions[ps]=find_flat(array_ps[:,-1], labels_ps)
#                plt.axvline(all_regions[ps][0][0],color='green')
#                plt.axvline(all_regions[ps][-1][1],color='green')
#                if len(all_regions[ps])>0:
#                    for j in range(len(all_regions[ps])):
#                        plt.axvline(all_regions[ps][j][0],color='green')
#                        plt.axvline(all_regions[ps][j][1],color='green')
    plt.figure(i)
    plt.savefig(perf_dir+node+'/'+str(int(ps))+'_''.png',bbox_inches='tight')
        i=i+1
                
                
ps1=1e4
ps2=27000

array_ps1=new_array[new_array[:,0]==ps1][:,:-1]
labels_ps1=new_labels[new_labels[:,0]==ps1][:,-1]

a_s=np.argsort(array_ps1[:,5])
for ir in range(np.shape(array_ps1)[1]):
    array_ps1[:,ir]=array_ps1[a_s,ir]
labels_ps1=labels_ps1[a_s] 


array_ps2=new_array[new_array[:,0]==ps2][:,:-1]
labels_ps2=new_labels[new_labels[:,0]==ps2][:,-1]

a_s=np.argsort(array_ps2[:,5])
for ir in range(np.shape(array_ps2)[1]):
    array_ps2[:,ir]=array_ps2[a_s,ir]
labels_ps2=labels_ps2[a_s]     



plt.figure(i)
plt.axes([0, 0, 2, 1])


g=np.arange(1,10).tolist()
k=10
target=11000
while k<=target:
    for j in range(1,10):
        if k*j<=target:
            g.append(k*j)
        else:
            break        
    k=k*10
if target not in g:
    g.append(target)


plt.scatter(ps1/array_ps1[:,5], labels_ps1/ps1,marker='.',label='ps:'+str(int(ps1)))   
plt.scatter(ps2/array_ps1[:,5], labels_ps1/ps2,marker='.',label='ps2 from '+str(int(ps1))) 
plt.scatter(ps1/array_ps1[:,5], labels_ps1/ps1,marker='.',label='ps2 from '+str(int(ps1)))   
  
plt.scatter(ps2/array_ps2[:,5], labels_ps2/ps2,marker='.',label='ps:'+str(int(ps2)))   

plt.xlabel('grain size')
plt.ylabel('execution time')
plt.xscale('log')
plt.title(str(th)+' threads')
plt.grid(True,'both')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



set1=set(array_ps1[:,5].tolist())
set2=set(array_ps2[:,5].tolist())
all_gs=[s for s in set1.intersection(set2)]
all_gs.sort()
all_gs=np.asarray(all_gs)

plt.figure(i)
for j in np.arange(np.shape(all_gs)[0]):
    plt.scatter(all_gs[j],labels_ps1[np.where(array_ps1[:,5]==all_gs[j])[0][0]]-labels_ps2[np.where(array_ps2[:,5]==all_gs[j])[0][0]])
#plt.scatter(all_gs, labels_ps1-labels_ps2,marker='.',label='ps:'+str(int(ps)))   
plt.xlabel('grain size')
plt.ylabel('execution time')
plt.xscale('log')
plt.title(str(th)+' threads')
plt.grid(True,'both')
plt.axvline(ps/th,color='gray',linestyle='dotted')  
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            


ps_ref=10000
