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


def create_dict(directories,to_csv=False):
    thr=[]
    data_files=[]
    
    for directory in directories:
        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]
    
    chunk_sizes=[]
    num_iterations=[]
    iter_lengths=[]
    nodes=[]
    
    if to_csv:
        f_csv=open('/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv','w')
        f_writer=csv.writer(f_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time'])

    for filename in data_files:
        if 'seq' in filename:
                (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                chunk_size=0
                th=1      
        elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
            (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=1
        else:
            (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
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
    
#    d={}
#    for node in nodes:
#        d[node]={}
#        for ni in num_iterations:              
#            d[node][ni]={}           
#            for th in thr:
#                d[node][ni][th]={}
#                for c in chunk_sizes:
#                    d[node][ni][th][c]={}
#                    for il in iter_lengths:
#                        d[node][ni][th][c][il]={}
                                                           
    data_files.sort()   
    problem_sizes=[]
    marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
    excludes=[marvin_dir+'/marvin_grain_size_1_1_2000_50000.dat',marvin_dir+'/marvin_grain_size_1_1_1000_100000.dat']
    
    for filename in data_files:   
        if filename not in excludes:             
            f=open(filename, 'r')
                     
            result=f.readlines()
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
            chunk_size=float(chunk_size)        
            th=float(th)       
            iter_length=float(iter_length)
            num_iteration=float(num_iteration)    
            first=True
            for r in [r for r in result if r!='\n']:  
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
results_dir='/home/shahrzad/repos/Blazemark/results/grain_size'
create_dict([marvin_dir,medusa_dir],1)


def my_func_g(ndata,alpha,gamma,d,h,q): 
    kappa=0.
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ts=ndata[:,0]
    return q*(N-1)*(N-2)+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d*(N-1)/N)*((n_t-1))*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)


def my_func_g(ndata,alpha,q): 
    kappa=0.
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ts=ndata[:,0]
    return q+alpha*L+(w_c)


def my_func_g(ndata,alpha,gamma,d,h,q): 
    kappa=0.
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ts=ndata[:,0]
    return alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*np.heaviside(n_t-N,1)+(d*(N-1)/N)*((n_t-1))*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)

def my_func_g(ndata,alpha,gamma,d,h,q,j): 
    kappa=0.
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    chunk_size=ndata[:,3]
    ts=ndata[:,0]
    return j*chunk_size+q*N+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d*(N-1)/N)*((n_t-1))*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)


titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'


dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()

problem_sizes=dataframe['problem_size'].drop_duplicates().values
problem_sizes.sort()


for node in nodes:
    node_selected=dataframe['node']==node
    df_n_selected=dataframe[node_selected][titles[1:]]
    
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()

    for ps in problem_sizes:
        ps_selected=dataframe['problem_size']==ps
        df_nps_selected=df_n_selected[ps_selected]
        
        ps_selected=dataframe['problem_size']==ps
        df_nps_selected=df_n_selected[ps_selected]
        
        array=df_nps_selected.values
        array=array.astype(float)
        
        data_size=int(np.shape(array)[0])
        print(int(ps),data_size)
        
        train_size=int(0.6*np.ceil(data_size*0.6))
        tes_size=data_size-train_size
        
        per = np.random.permutation(data_size)
        train_set=array[per[0:train_size],:-1] 
        train_labels=array[per[0:train_size],-1]  
        test_set=array[per[train_size:],:-1]  
        test_labels=array[per[train_size:],-1]  
        
        param_bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,np.inf])
#        param_bounds=([0,-np.inf],[np.inf,np.inf])

        popt, pcov=curve_fit(my_func_g,train_set,train_labels,method='trf',bounds=param_bounds)

        
        i=1
        for th in thr:
            new_array=train_set[train_set[:,2]==th]

            plt.figure(i)
#            z=my_func_g(new_array,*popt)

            plt.scatter(new_array[:,5],train_labels[train_set[:,2]==th],marker='.',label='true')
            plt.xlabel('grain size')
            plt.ylabel('execution time')
            plt.xscale('log')
            plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
            plt.axvline(ps/th,color='gray',linestyle='dotted')  
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            i=i+1
            
            
        i=1
        for th in thr:
            new_array=test_set[test_set[:,2]==th]

            plt.figure(i)
            z=my_func_g(new_array,*popt)

            plt.scatter(new_array[:,5],test_labels[test_set[:,2]==th]/ps,marker='.',label='true')
            plt.scatter(new_array[:,5],z/ps,marker='.',label='pred')

            plt.xlabel('grain size')
            plt.ylabel('execution time')
            plt.xscale('log')
            plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
            plt.axvline(ps/th,color='gray',linestyle='dotted')  
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            i=i+1
            
                

        
        
def find_fit(ndata, labels, ncols=5):
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=(M-1)*w_c
    Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
    Q[:,3]=n_t*np.heaviside(n_t-N,1)
    Q[:,4]=N

    m,_ = nnls(Q, labels-w_c)
    return m

def find_val(ndata, model):
    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=(M-1)*w_c
    Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
    Q[:,3]=n_t*np.heaviside(n_t-N,1)
    Q[:,4]=N
    return np.dot(Q,model)+w_c



model=find_fit(train_set,train_labels)        
i=1
for th in thr:
    new_array=test_set[test_set[:,2]==th]

    plt.figure(i)
    z1=find_val(new_array,model)
    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],ps/test_labels[test_set[:,2]==th],marker='.',label='true')
    plt.scatter(new_array[:,5],ps/z1,marker='.',label='pred')
    plt.scatter(new_array[:,5],ps/z2,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('mflops')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1
    
i=1
for th in thr:
    new_array=test_set[test_set[:,2]==th]

    plt.figure(i)
    z1=find_val(new_array,model)
    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],test_labels[test_set[:,2]==th]/ps,marker='.',label='true')
    plt.scatter(new_array[:,5],z1/ps,marker='.',label='pred')
    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('execution time')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1
    
    
###################################################################################################
# taking number of threads out
def find_fit(ndata, labels, params=None,ncols=4,step=0):
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    

    if step==0:
        Q=np.zeros((np.shape(ndata)[0],ncols))
        Q[:,0]=L
        Q[:,1]=w_c*np.heaviside(N-1,0)
        Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
        Q[:,3]=n_t
#        Q[:,4]=1
#        Q[:,5]=1
        m,_ = nnls(Q, labels-w_c)
        return m
    else:
        if step==1:
            Q=np.zeros((np.shape(ndata)[0],ncols-1))
            Q[:,0]=w_c*np.heaviside(N-1,0)
            Q[:,1]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
            Q[:,2]=n_t
    #        Q[:,3]=N-1
            m,_ = nnls(Q, labels-w_c-L*params[0])
    #        return np.asarray([params[0],m[0],m[1],m[2],m[3],params[-1]])
            return np.asarray([params[0],m[0],m[1],m[2]])
        else:
            Q=np.zeros((np.shape(ndata)[0],ncols-2))
            Q[:,0]=w_c*np.heaviside(N-1,0)
            Q[:,1]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
#            Q[:,2]=1
    #        Q[:,3]=N-1
            m,_ = nnls(Q, labels-w_c-L*params[0]-n_t*params[-1])
    #        return np.asarray([params[0],m[0],m[1],m[2],m[3],params[-1]])
            return np.asarray([params[0],m[0],m[1],params[-1]])

def find_val(ndata, model):
    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=w_c*np.heaviside(N-1,0)
    Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
    Q[:,3]=n_t
#    Q[:,4]=1
#    Q[:,5]=1
    return np.dot(Q,model)+w_c

ps_selected=dataframe['problem_size']==ps
df_n_selected=dataframe[node_selected][titles[1:]]
df_nps_selected=df_n_selected[ps_selected]

array=df_nps_selected.values
array=array.astype(float)

i=1
th=1
array_th=array[array[:,2]==th]
a_s=np.argsort(array_th[:,5])
array_th=array_th[a_s]
  
data_size=int(np.shape(array_th)[0])
print(int(ps),data_size)

train_size=data_size#int(np.ceil(data_size*0.6))
tes_size=data_size-train_size

per = np.random.permutation(data_size)
train_set=array_th[per[0:train_size],:-1] 
train_labels=array_th[per[0:train_size],-1]  
test_set=array_th[per[train_size:],:-1]  
test_labels=array_th[per[train_size:],-1]  

model=find_fit(train_set,train_labels)      
alpha=model[0]
q=model[-1]


params=[]
 
params.append(model)

new_array=train_set

plt.figure(i)
z1=find_val(new_array,model)
#z2=my_func_g(new_array,*popt)
plt.scatter(new_array[:,5],train_labels/ps,marker='.',label='true')
plt.scatter(new_array[:,5],z1/ps,marker='.',label='pred')
#    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

plt.xlabel('grain size')
plt.ylabel('execution time')
plt.xscale('log')
plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
plt.axvline(ps/th,color='gray',linestyle='dotted')  
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

i=i+1
    
th=2

for th in thr[1:]:
    array_th=array[array[:,2]==th]
    data_size=int(np.shape(array_th)[0])
    print(int(ps),data_size)
    
    train_size=data_size#int(np.ceil(data_size*0.6))
    tes_size=data_size-train_size
    
    per = np.random.permutation(data_size)
    train_set=array_th[per[0:train_size],:-1] 
    train_labels=array_th[per[0:train_size],-1]  
    test_set=array_th[per[train_size:],:-1]  
    test_labels=array_th[per[train_size:],-1]  
    model=find_fit(train_set,train_labels,params[int(th-2)],step=int(th)-1)        

#    model=find_fit(train_set,train_labels,params[1],step=2)        
    params.append(model)
    
    new_array=train_set

    plt.figure(i)
    z1=find_val(new_array,model)
#    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],train_labels/ps,marker='.',label='true')
    plt.scatter(new_array[:,5],z1/ps,marker='.',label='pred')
#    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('execution time')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1
    plt.figure(i)
    z1=find_val(new_array,model)
    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],ps/train_labels,marker='.',label='true')
    plt.scatter(new_array[:,5],ps/z1,marker='.',label='pred')
#    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('mflops')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1
    
    
labels=['alpha','gamma','d','h','q']
for i in range(len(params[0])):    
    plt.figure(i+1)
    plt.scatter(thr,[p[i] for p in params],label=labels[i])    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.scatter(thr,[p[0]+p[-2] for p in params],label=labels[i])    

#############################################################



def find_fit(ndata, labels, params=None,ncols=4,step=0):
    ps=ndata[:,0]
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    

    if step==0:
        Q=np.zeros((np.shape(ndata)[0],ncols))
        Q[:,0]=L
        Q[:,1]=w_c*np.heaviside(N-1,0)
        Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
        Q[:,3]=n_t*np.heaviside(n_t-N,1)
#        Q[:,4]=N-1
#        Q[:,5]=1
        m,_ = nnls(Q, labels-w_c)
        return m

def find_val(ndata, model):
    ps=ndata[:,0]

    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=w_c*np.heaviside(N-1,0)
    Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
    Q[:,3]=n_t*np.heaviside(n_t-N,1)
#    Q[:,4]=N-1
#    Q[:,5]=1
    return np.dot(Q,model)+w_c

node_selected=dataframe['node']==node
df_n_selected=dataframe[node_selected][titles[1:]]

thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()

array=df_n_selected.values
array=array.astype(float)

th=1
array_th=array[array[:,2]==th]
a_s=np.argsort(array_th[:,5])
array_th=array_th[a_s]

data_size=int(np.shape(array_th)[0])
print(data_size)

train_size=data_size#int(np.ceil(data_size*0.6))
tes_size=data_size-train_size

per = np.random.permutation(data_size)
train_set=array_th[per[0:train_size],:-1] 
train_labels=array_th[per[0:train_size],-1]  
test_set=array_th[per[train_size:],:-1]  
test_labels=array_th[per[train_size:],-1]  

model=find_fit(train_set,train_labels)   

for ps in problem_sizes:
    new_array=array_th[array_th[:,0]==ps][:,:-1]
    plt.figure(i)
    z1=find_val(new_array,model)
#    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],array_th[array_th[:,0]==ps][:,-1]/ps,marker='.',label='true')
    plt.scatter(new_array[:,5],z1/ps,marker='.',label='pred')
#    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('execution time')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1

##########################################################################
    #data: 1 thread 1 task
##########################################################################
def find_fit(ndata, labels, ncols=2,step=0):
    ps=ndata[:,0]
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    chunk_size=ndata[:,3]

    if step==0:
        Q=np.zeros((np.shape(ndata)[0],ncols))
        Q[:,0]=L        
        Q[:,1]=chunk_size
        m,_ = nnls(Q, labels-w_c)
        return m
    else:
        alpha=31.5456
        Q=np.zeros((np.shape(ndata)[0],ncols-1))
        Q[:,0]=chunk_size
        m,_ = nnls(Q, labels-w_c-L*alpha)
        return np.asarray([alpha, m[0]])

def find_val(ndata, model):
    ps=ndata[:,0]
    chunk_size=ndata[:,3]

    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=chunk_size
#    Q[:,5]=1
    return np.dot(Q,model)+w_c


node_selected=dataframe['node']==node
df_n_selected=dataframe[node_selected][titles[1:]]


nt_selected=dataframe['num_tasks']==1
df_n_selected=dataframe[node_selected & nt_selected][titles[1:]]

thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()

array=df_n_selected.values
array=array.astype(float)

th=1
array_th=array[array[:,2]==th]

data_size=int(np.shape(array_th)[0])
print(data_size)

models={}
first=1
for ps in problem_sizes:

#    print(ps)
    
    train_set=array_th[array_th[:,0]==ps][:,:-1]
    train_labels=array_th[array_th[:,0]==ps][:,-1]  
    
    #test_set=array_th[per[train_size:],:-1]  
    #test_labels=array_th[per[train_size:],-1]  
    if np.shape(train_set)[0]>0:
        if first:
            step=0
            first=0
        else:
            step=1
        model=find_fit(train_set,train_labels,step=0)   
        if sum(model)!=0:
            models[ps]=(model)
#            new_array=array_th[array_th[:,0]==ps][:,:-1]
#            plt.figure(i)
#            z1=find_val(new_array,model)
#        #    z2=my_func_g(new_array,*popt)
#            plt.scatter(new_array[:,5],array_th[array_th[:,0]==ps][:,-1]/ps,marker='.',label='true')
#            plt.scatter(new_array[:,5],z1/ps,marker='.',label='pred')
#        #    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')
#        
#            plt.xlabel('grain size')
#            plt.ylabel('execution time')
#            plt.xscale('log')
#            plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
#            plt.axvline(ps/th,color='gray',linestyle='dotted')  
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        
#            i=i+1
    
    
plt.scatter([m for m in models.keys()],[models[m][0] for m in models.keys()])
plt.xscale('log')
plt.scatter([m for m in models.keys()],[models[m][1] for m in models.keys()])


#y=np.asarray([models[m][1] for m in models.keys()])
#x=np.asarray([m for m in models.keys()])
#
#n=np.polyfit(x,y,1)
#p=np.poly1d(n)
#p(1e7)
##########################################################################
#sequential
##########################################################################
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hxp_for_loop/'
create_dict([marvin_dir,medusa_dir],1)
titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'


dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

node_selected=dataframe['node']==node
cs_selected=dataframe['chunk_size']==0
th_selected=dataframe['num_threads']==1

df_n_selected=dataframe[node_selected & th_selected][titles[1:]]

#df_n_selected=dataframe[node_selected][titles[1:]]

thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()

array=df_n_selected.values
array=array.astype(float)

data_size=int(np.shape(array)[0])
print(data_size)

array_seq=array[array[:,3]==0]
i=1
plt.figure(i)
plt.axes([0, 0, 3, 1])

for ii in range(np.shape(array)[0]):    
    plt.scatter(array_seq[ii,1],array_seq[ii,-1]-array_seq[ii,0],marker='.')
    
    plt.xlabel('problem size')
    plt.ylabel('overhead(microseconds)')
    plt.title('sequential execution time')

problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
problem_sizes.sort()
 
n=np.polyfit(array_seq[:,1],array_seq[:,-1]-array_seq[:,0],1)    
p=np.poly1d(n)   

array_cs_1=array[array[:,3]==1]
array_cs_nt_1=array_cs_1[array_cs_1[:,7]==1]

plt.figure(i)
for ps in problem_sizes[-10:]:
#    plt.figure(i)
    array_ps=array_seq[array_seq[:,0]==ps]
    array_ps_cs=array_cs_nt_1[array_cs_nt_1[:,0]==ps]
    plt.scatter(array_ps[:,1],array_ps[:,-1]-ps,marker='.',color='blue',label='true')
#    plt.scatter(array_ps_cs[:,1],array_ps_cs[:,-1]-ps,marker='.',color='red',label='one task')
    plt.scatter(array_ps[:,1],p(array_ps[:,1]),marker='.',color='green',label='fit')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('num iterations')
    plt.xscale('log')
    plt.ylabel('overhead(microseconds)')
plt.figure(i)
plt.savefig(perf_dir+'overheads_seq.png',bbox_inches='tight')
    plt.title('problem size '+str(int(ps)))
    i=i+1
    plt.axhline(ps)
    

plt.figure(i)
problem_sizes=list(set(array_seq[:,0]))
problem_sizes.sort()


for ps in problem_sizes[-10:]:
#    plt.figure(i)
    array_ps=array_seq[array_seq[:,0]==ps]
    array_ps_cs=array[array[:,0]==ps]
    array_ps_cs=array_ps_cs[array_ps_cs[:,7]==1]
    if np.shape(array_ps_cs)[0]>0:
        for ni in set(array_ps_cs[:,1]):
            plt.scatter(ni,array_ps_cs[array_ps_cs[:,1]==ni][:,-1]-array_ps[array_ps[:,1]==ni][:,-1],marker='.',label=ni)
        #    plt.scatter(array_ps_cs[:,1],array_ps_cs[:,-1]-ps,marker='.',color='red',label='one task')
        #    plt.scatter(array_ps[:,1],p(array_ps[:,1]),marker='.',color='green',label='fit')
        
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.xlabel('num iterations')
            plt.xscale('log')
            plt.ylabel('overhead(microseconds)')
##########################################################################
 #1-new   #data: 1 thread 1 task chunk_size=1
##########################################################################
create_dict([marvin_dir,medusa_dir],1)
titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'


dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)
    
from scipy.optimize import lsq_linear
def find_fit(ndata, labels, alpha=27.7, step=1):
    ncols=1
    N=ndata[:,2]
    ps=ndata[:,0]
    n_t=ndata[:,7]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,6]
    chunk_size=ndata[:,3]
    num_iter=ndata[:,1]

    il=ndata[:,5]

    if step==0:
        Q=np.zeros((np.shape(ndata)[0],ncols))
        Q[:,0]=L  
        m,_ = nnls(Q, labels-w_c)
        return m
    else:
        Q=np.zeros((np.shape(ndata)[0],ncols+1))
        Q[:,0]=chunk_size
        Q[:,1]=1
        m,_ = nnls(Q, labels-w_c-alpha*np.heaviside(chunk_size,0))
        return np.array([alpha,m[0],m[1]])

def find_val(ndata, model):
    chunk_size=ndata[:,3]
    ps=ndata[:,0]

    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,7]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,6]
    il=ndata[:,5]
    num_iter=ndata[:,1]

    Q[:,0]=np.heaviside(chunk_size,0)
    Q[:,1]=chunk_size 
    Q[:,2]=1
#    Q[:,5]=1
    return np.dot(Q,model)+w_c


node_selected=dataframe['node']==node
nt_selected=dataframe['num_tasks']<=1
cs_selected=dataframe['chunk_size']<=1

df_n_selected=dataframe[node_selected & nt_selected & cs_selected][titles[1:]]

#df_n_selected=dataframe[node_selected][titles[1:]]

thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()

array=df_n_selected.values
array=array.astype(float)

th=1
array_th=array[array[:,2]==th]

a_s=np.argsort(array_th[:,0])
array_th=array_th[a_s]

#alpha=array_th[0,-1]-array_th[0,0]
alpha=0
data_size=int(np.shape(array_th)[0])
print(data_size)
train_size=int(0.6*data_size)
test_size=data_size-train_size
per = np.random.permutation(data_size)

train_set=array_th[per[0:train_size],:-1]
train_labels=array_th[per[0:train_size],-1]  
model1=find_fit(train_set,train_labels,step=1)   

i=1
plt.figure(i)
plt.axes([0, 0, 3, 1])

for ii in range(np.shape(array_th)[0]):    
    z1=find_val(np.matrix(array_th[ii,:-1]),model1)
    plt.scatter(ii,array_th[ii,-1]-array_th[ii,0],marker='.',color='blue')
    plt.scatter(ii,z1[0,0]-array_th[ii,0],marker='.',color='red')
    
    plt.xlabel('data point')
    plt.ylabel('overhead(microseconds)')
    plt.title('1 task 1 threads with chunk_size=1')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
               
#    plt.annotate(str(int(array_th[ii,0])), # this is the text
#                 (ii,array_th[ii,-1]-array_th[ii,0]), # this is the point to label
#                 textcoords="offset points", # how to position the text
#                 xytext=(0,10), # distance from text to points (x,y)
#                 ha='center') # horizontal alignment can be left, right or center
 
    i=i+1 

i=1
plt.figure(i)
plt.axes([0, 0, 2, 1])

for ii in range(np.shape(array_th)[0]):    
    z1=find_val(np.matrix(array_th[ii,:-1]),model1)
    plt.scatter(array_th[ii,0],(array_th[ii,-1]-array_th[ii,0]),marker='.',color='blue')
    plt.scatter(array_th[ii,0],z1[0,0]-array_th[ii,0],marker='.',color='red')
    plt.xscale('log')
    plt.xlabel('problem size')
    plt.ylabel('overhead(microseconds)')
    plt.title('1 task 1 threads with chunk_size=1')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)                 
 
    i=i+1 
    
##########################################################################
 #1-new   #data: 1 thread 1 task 
##########################################################################
from scipy.optimize import lsq_linear
def find_fit(ndata, labels, alpha=0, step=1):
    ncols=2
    N=ndata[:,2]
    n_t=ndata[:,7]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,6]
    iter_length=ndata[:,4]    
    num_blocks=ndata[:,1]
    
    if step==0:
        Q=np.zeros((np.shape(ndata)[0],ncols))
        Q[:,0]=L*w_c        
        Q[:,1]=num_blocks
        m,_ = nnls(Q, labels-w_c)
        return m

def find_val(ndata, model):
    chunk_size=ndata[:,3]

    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,7]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,6]
    iter_length=ndata[:,4]    
    num_blocks=ndata[:,1]
    
    Q[:,0]=L*w_c
    Q[:,1]=num_blocks

#    Q[:,5]=1
    return np.dot(Q,model)+w_c


node_selected=dataframe['node']==node
nt_selected=dataframe['num_tasks']==1
df_n_selected=dataframe[node_selected & nt_selected][titles[1:]]

#df_n_selected=dataframe[node_selected][titles[1:]]

thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()

array=df_n_selected.values
array=array.astype(float)

th=1
array_th=array[array[:,2]==th]

a_s=np.argsort(array_th[:,0])
array_th=array_th[a_s]

#alpha=array_th[0,-1]-array_th[0,0]
alpha=0
data_size=int(np.shape(array_th)[0])
print(data_size)
train_size=int(0.6*data_size)
test_size=data_size-train_size
per = np.random.permutation(data_size)

train_set=array_th[per[0:train_size],:-1]
train_labels=array_th[per[0:train_size],-1]  
model1=find_fit(train_set,train_labels,alpha=alpha,step=0)   

i=1
plt.figure(i)
plt.axes([0, 0, 2, 1])

for ii in range(np.shape(array_th)[0]):    
    z1=find_val(np.matrix(array_th[ii,:-1]),model1)
    plt.scatter(ii,array_th[ii,-1]-array_th[ii,6],marker='.',color='blue')
    plt.scatter(ii,z1[0,0]-array_th[ii,0],marker='.',color='red')
    
    plt.xlabel('data point')
    plt.ylabel('execution time')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    i=i+1 

##########################################################################
#2-data: 1 thread >1 task
##########################################################################    
def find_fit(ndata, labels):
    ncols=2
    ps=ndata[:,0]
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    chunk_size=ndata[:,3]

    Q=np.zeros((np.shape(ndata)[0],ncols))
    Q[:,0]=L/ps
    Q[:,1]=1/ps
    m,_=nnls(Q, labels-w_c/ps)
    return m

def find_val(ndata, model):
    ps=ndata[:,0]
    chunk_size=ndata[:,3]

    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L/ps
    Q[:,1]=1/ps
    return np.dot(Q,model)+w_c/ps


node_selected=dataframe['node']==node
df_n_selected=dataframe[node_selected][titles[1:]]

array=df_n_selected.values
array=array.astype(float)

th=1
array_th=array[array[:,2]==th]

a_s=np.argsort(array_th[:,0])
array_th=array_th[a_s]

data_size=int(np.shape(array_th)[0])

train_set=array_th[:,:-1]
train_labels=array_th[:,-1]/array_th[:,0]  
model2=find_fit(train_set,train_labels)   

i=1
for ps in problem_sizes[-100:]:
    train_set=array_th[array_th[:,0]==ps][:,:-1]
    train_labels=array_th[array_th[:,0]==ps][:,-1]  
    if np.shape(np.unique(train_set[:,5]))[0]>10 and np.shape(train_set)[0]>20:
        plt.figure(i)
        z1=find_val(train_set,model2)
        plt.scatter(array_th[array_th[:,0]==ps][:,5],array_th[array_th[:,0]==ps][:,-1]/ps,marker='.',label='true')
        plt.scatter(array_th[array_th[:,0]==ps][:,5],z1,marker='.',label='pred')
    
        plt.xlabel('grain size')
        plt.ylabel('execution time')
        plt.xscale('log')
        plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
        plt.axvline(ps/th,color='gray',linestyle='dotted')  
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
        i=i+1 
      
    
##########################################################################
#3-data: 1> thread >1 task for a fixed problem size where n_t=th, how does gamma change?
##########################################################################    
def find_fit(ndata, labels, model):
    ncols=2
    ps=ndata[:,0]
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    chunk_size=ndata[:,3]

    Q=np.zeros((np.shape(ndata)[0],ncols-1))
    Q[:,0]=w_c*np.heaviside(M-1,0)*(M-1)
    m,_ = nnls(Q, labels-w_c-L*model[0]-chunk_size*model[1]-n_t*model[2])
    return np.asarray([model[0],model[1],model[2],m[0]])

def find_val(ndata, model):
    ps=ndata[:,0]
    chunk_size=ndata[:,3]
    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=chunk_size
    Q[:,2]=n_t
    Q[:,3]=w_c*np.heaviside(M-1,0)*(M-1)
    return np.dot(Q,model)+w_c


node_selected=dataframe['node']==node
df_n_selected=dataframe[node_selected][titles[1:]]

array=df_n_selected.values
array=array.astype(float)
train_set=array[:,:-1]
train_labels=array[:,-1] 
model3=find_fit(train_set,train_labels,model2)   
i=1
for ps in problem_sizes:
    array_ps=array[array[:,0]==ps]

    for th in thr:
        plt.figure(i)        
        array_th=array_ps[array_ps[:,2]==th]
        if np.shape(array_th)[0]>20:
            z1=find_val(array_th[:,:-1],model3)
            
            plt.scatter(array_th[:,5],array_th[:,-1]/ps,marker='.',label='true')
            plt.scatter(array_th[:,5],z1/ps,marker='.',label='pred')
            plt.xscale('log')        
    
            plt.xlabel('num threads')
            plt.ylabel('execution time')
            plt.title('problem size:'+str(ps)+' '+str(int(th))+' threads')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
            plt.axvline(ps/th,color='gray',linestyle='dotted')  

            i=i+1



models={}
for ps in problem_sizes:
    array_ps=array[array[:,0]==ps]
    array_nt=array_ps[array_ps[:,2]==array_ps[:,-2]]
    array_c=array_nt[array_nt[:,0]==array_nt[:,2]*array_nt[:,3]]
    train_set=array_nt[array_nt[:,0]==array_nt[:,2]*array_nt[:,3]][:,:-1]
    train_labels=array_nt[array_nt[:,0]==array_nt[:,2]*array_nt[:,3]][:,-1]
    if np.shape(train_labels)[0]>2:
        model3=find_fit(train_set,train_labels,model2)   
        models[ps]=model3

        plt.figure(i)
        z1=find_val(train_set,model3)

        plt.scatter(array_nt[array_nt[:,0]==array_nt[:,2]*array_nt[:,3]][:,2],train_labels/array_nt[array_nt[:,0]==array_nt[:,2]*array_nt[:,3]][:,0],marker='.',label='true')
        plt.scatter(array_nt[array_nt[:,0]==array_nt[:,2]*array_nt[:,3]][:,2],z1/array_nt[array_nt[:,0]==array_nt[:,2]*array_nt[:,3]][:,0],marker='.',label='pred')

        plt.xlabel('num threads')
        plt.ylabel('execution time')
        plt.title('problem size:'+str(ps)+' 1 task per core created')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
    i=i+1 
        
        if np.shape(train_set)[0]>2:
            plt.figure(i)
            plt.scatter(array[array[:,0]==ps][:,2],array[array[:,0]==ps][:,-1]/ps,marker='.',label='true')
        
            plt.xlabel('num threads')
            plt.ylabel('execution time')
            plt.title('problem size:'+str(ps)+' 1 task created')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
            i=i+1 
            model=find_fit(train_set,train_labels,model)   
            models[ps]=model
            
plt.scatter([m for m in models.keys()],[models[m][-1] for m in models.keys()])
plt.xscale('log')        
plt.xlabel('problem size')
plt.ylabel('gamma')        
plt.title(' 1 task per core (same size) created')


#############################################
#for a fixed problem size where n_t=th, how does gamma change?
def find_fit(ndata, labels, model):
    ncols=2
    ps=ndata[:,0]
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]*np.heaviside(N-1,0)
    chunk_size=ndata[:,3]

    Q=np.zeros((np.shape(ndata)[0],ncols-1))
    Q[:,0]=w_c
    m, _, _, _ = np.linalg.lstsq(Q, labels-w_c-L*model[0]-chunk_size*model[1]-n_t*model[2])
    return np.asarray([model[0],model[1],model[2],m[0]])

def find_val(ndata, model):
    ps=ndata[:,0]
    chunk_size=ndata[:,3]

    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=chunk_size
    Q[:,2]=n_t
    Q[:,3]=w_c*np.heaviside(N-1,0)
    return np.dot(Q,model)+w_c


node_selected=dataframe['node']==node
nt_selected=dataframe['num_tasks']==1
df_n_selected=dataframe[node_selected & nt_selected][titles[1:]]

array=df_n_selected.values
array=array.astype(float)

models={}
for ps in problem_sizes:
    train_set=array[array[:,0]==ps][:,:-1]
    train_labels=array[array[:,0]==ps][:,-1]  
    
    if np.shape(train_set)[0]>2:
        plt.figure(i)
        plt.scatter(array[array[:,0]==ps][:,2],array[array[:,0]==ps][:,-1]/ps,marker='.',label='true')
    
        plt.xlabel('num threads')
        plt.ylabel('execution time')
        plt.title('problem size:'+str(ps))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
        i=i+1 
        model=find_fit(train_set,train_labels,model)   
        models[ps]=model
        
plt.scatter([m for m in models.keys()],[models[m][-1] for m in models.keys()])
plt.xscale('log')        
plt.xlabel('problem size')
plt.ylabel('gamma')        

##########################################
#n_t=1
def find_fit(ndata, labels, params=None,ncols=5,step=0):
    ps=ndata[:,0]
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    

    if step==0:
        Q=np.zeros((np.shape(ndata)[0],ncols))
        Q[:,0]=L
        Q[:,1]=w_c*np.heaviside(N-1,0)
        Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
        Q[:,3]=n_t
        Q[:,4]=ps
#        Q[:,5]=1
        m,_ = nnls(Q, labels-w_c)
        return m

def find_val(ndata, model):
    ps=ndata[:,0]

    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=w_c*np.heaviside(N-1,0)
    Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
    Q[:,3]=n_t
    Q[:,4]=ps
#    Q[:,5]=1
    return np.dot(Q,model)+w_c

node_selected=dataframe['node']==node
nt_selected=dataframe['num_tasks']==1
df_n_selected=dataframe[node_selected & nt_selected][titles[1:]]

thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()

array=df_n_selected.values
array=array.astype(float)

th=1
array_th=array[array[:,2]==th]

data_size=int(np.shape(array_th)[0])
print(data_size)

models={}

#    print(ps)
    
train_set=array_th[:,:-1]
train_labels=array_th[:,-1]  

#test_set=array_th[per[train_size:],:-1]  
#test_labels=array_th[per[train_size:],-1]  
#model=find_fit(train_set,train_labels)   

plt.figure(i)
z1=find_val(train_set,model)
#    z2=my_func_g(new_array,*popt)
plt.scatter(array_th[:,0],array_th[:,-1]-array_th[:,0],marker='.',label='true')
#plt.scatter(array_th[:,0],z1,marker='.',label='pred')
#    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

plt.xlabel('problem size(microseconds)')
plt.ylabel('overhead(microseconds)')
plt.xscale('log')
plt.title('1 thread, 1 task')
#plt.axvline(ps/th,color='gray',linestyle='dotted')  
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

i=i+1

problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
problem_sizes.sort()

all_n=[]
for ps in problem_sizes:
    plt.figure(i)
    array_ps=array_th[array_th[:,0]==ps]
    if np.shape(array_ps)[0]>10:

        plt.scatter(array_ps[:,3],array_ps[:,-1]-array_ps[:,0]-array_ps[:,6],marker='.',label='true')
#        n=np.polyfit((array_ps[:,3]),(array_ps[:,-1]-ps),1)
#        all_n.append(n)
#        p=np.poly1d(n)
#        plt.scatter(array_ps[:,3],(p(array_ps[:,3])),marker='.',label='pred')

        plt.xlabel('chunk_size')
        plt.ylabel('overhead(microseconds)')
        plt.xscale('log')
        plt.title('1 thread, 1 task')
        #plt.axvline(ps/th,color='gray',linestyle='dotted')  
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('problem size:'+str(int(ps)))
        i=i+1