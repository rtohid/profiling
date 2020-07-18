#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:56:46 2018

@author: shahrzad
"""

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
import math

static_date_str='10-13-18-2224'
dynamic_date_str='11-08-18-1330'
directory='/home/shahrzad/Spyder/Blazemark/results/data'
openmp_dir=directory+'/openmp/10-22-18-1554'
dynamic_dir=directory+'/dynamic/'+dynamic_date_str
static_dir=directory+'/static/'+static_date_str
perf_dir=directory+'/performance plots'

from matplotlib.backends.backend_pdf import PdfPages
data_files=glob.glob(dynamic_dir+'/*.dat')
    
thr=[]
sizes=[]
benchmarks=[]
block_sizes=[]
chunk_sizes=[]
for filename in data_files:
    if 'hpx' in filename:
        file_str=filename.split('.')[0].split('/')[-1]
        (benchmark, th, runtime, chunk_size, block_size, size) = file_str.split('-')

        if int(th) not in thr:
            thr.append(int(th))
        if int(size) not in sizes:
            sizes.append(int(size))
        if benchmark not in benchmarks:
            benchmarks.append(benchmark)
        if int(block_size) not in block_sizes:
            block_sizes.append(int(block_size))
        if int(chunk_size) not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))

thr.sort()
sizes.sort()
benchmarks.sort()
block_sizes.sort()
d_all={}

for benchmark in benchmarks:   
    d_all[benchmark]={}
    for s in sizes:       
        d_all[benchmark][s]={}
        for cs in chunk_sizes:    
            d_all[benchmark][s][cs]={}
            for bs in block_sizes:
                d_all[benchmark][s][cs][bs]=[0]*len(thr)    

for filename in data_files:
    if 'hpx' in filename:
        (benchmark, cores, runtime, chunk_size, block_size, size) = filename.split('/')[-1].replace('.dat','').split('-')
        print(benchmark, cores, runtime, chunk_size, block_size, size)
    
        f=open(filename, 'r')
    
        result=f.readlines()[3]

        s=int(result.strip().split(' ')[0])
        if s!=int(size):
            print('error')
        print("s:",s)
        d_all[benchmark][s][int(chunk_size)][int(block_size)][int(cores)-1]=float(result.strip().split(' ')[-1])       


data_files=glob.glob(static_dir+'/*.dat')

thr=[]
vec_sizes=[]
block_sizes=[]
for filename in data_files:
    if 'hpx' in filename:
        vec_size =int(filename.split('-')[-1].split('.')[0])
        th =int(filename.split('-')[-4])
        block_size =int(filename.split('-')[-2])

        if th not in thr:
            thr.append(th)
        if vec_size not in vec_sizes:
            vec_sizes.append(vec_size)
        if block_size not in block_sizes:
            block_sizes.append(block_size)

thr.sort()
vec_sizes.sort()
block_sizes.sort()

d_static={}     

for benchmark in benchmarks:   
    d_static[benchmark]={}
    for s in sizes:       
        d_static[benchmark][s]={}
        for bs in block_sizes:
            d_static[benchmark][s][bs]=[0]*len(thr)    
                
for filename in data_files:
    f=open(filename, 'r')
    results=f.readlines()[3]
    if 'hpx' in filename:
        (benchmark, th, runtime, block_size, vec_size) = filename.split('/')[-1].replace('.dat','').split('-')
        print(benchmark, th, runtime, block_size, vec_size)
        
        time_result=float(results.split(str(vec_size))[1].strip())
        d_static[benchmark][int(vec_size)][int(block_size)][int(th)-1]=time_result
        
#data_files=glob.glob(openmp_dir+'/*.dat')
#
#openmp_benchmarks=[]
#d_openmp={}   
#for filename in data_files:
#    if 'openmp' in filename:
#        (repeat,benchmark, cores, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
#        if benchmark not in openmp_benchmarks:
#            openmp_benchmarks.append(benchmark)
#
#openmp_benchmarks.sort()            
#for benchmark in openmp_benchmarks:   
#    d_openmp[benchmark]={}
#    for s in sizes:
#        d_openmp[benchmark][s]=[0]*len(thr)
#        
#for filename in data_files:
#    if 'openmp' in filename:
#        (repeat,benchmark, cores, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
#        print(benchmark, cores, runtime)    
#    
#
#    f=open(filename, 'r')
#             
#    if 'openmp' in filename:        
#        f=open(filename, 'r')
#        result=f.readlines()[3:9]
#
#        for r in result:
#            s=int(r.strip().split(' ')[0])
#            print("s:",s)
#            if s in sizes:
#                d_openmp[benchmark][s][int(cores)-1]=float(r.strip().split(' ')[-1]) 
#                
        
import pickle
#################################
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
##################################    
d_openmp=load_obj(openmp_dir+'/openmp_benchmarks')

pp = PdfPages(perf_dir+'/perf_static_'+static_date_str+'_dynamic_'+dynamic_date_str+'.pdf')

i=1
for benchmark in benchmarks:   
    for s in sizes:        
        for k in d_all[benchmark][s].keys():   
            plt.figure(i)
            for b in d_all[benchmark][s][k].keys(): 
                plt.plot(thr, d_all[benchmark][s][k][b],label="chunk size="+str(k)+" block size="+str(b))  

                if b in d_static[benchmark][s].keys():
                    plt.plot(thr, d_static[benchmark][s][b],color='black',label="static block size="+str(b))  

                plt.xlabel('#number of cores')
                plt.ylabel('MFLop/s')
                plt.title(benchmark+'  '+static_date_str+' '+dynamic_date_str+'\nvector size: '+str(s))
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            i=i+1
    
            plt.plot(thr, d_openmp[benchmark][s],label='openmp', linestyle='--', color='black')
        
        
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        

#for benchmark in benchmarks:   
#    for s in sizes:
#        plt.figure(i)
#        for k in d_all[benchmark][s]:        
#            plt.plot(thr, d_all[benchmark][s][k],label="chunk size="+str(k))   
#        plt.plot(thr, d_static[s],label='static',color='black')
#    
#        plt.plot(thr, d_openmp[benchmark][s],label='openmp', linestyle='--', color='black')
#        plt.xlabel('#number of cores')
#        plt.ylabel('MFLop/s')
#        plt.title(benchmark+'  '+date_str+'\n\n\nvector size: '+str(s))
#        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        i=i+1
#        
#        plt.savefig(pp, format='pdf',bbox_inches='tight')
#        print('')
plt.show()
pp.close()    


    
#threshold=10
#pp = PdfPages(dynamic_dir+'/blazemark_hpx_chunk_size_'+str(threshold)+'.pdf')
#    
#i=1
#for benchmark in benchmarks:   
#    for s in sizes[-2:]:
#        for k in d_all[benchmark][s]:
#            if int(k)<(s/threshold):
#                plt.figure(i)
#                plt.plot(thr, d_all[benchmark][s][k],label="chunk size="+str(k))   
#            else:
#                plt.figure(i+1)
#                plt.plot(thr, d_all[benchmark][s][k],label="chunk size="+str(k))   
##            plt.plot(thr, d_static[benchmark][s],color='black')
#            plt.plot(thr, d_openmp[benchmark][s], linestyle='--', color='black')
#    
#            plt.savefig(pp, format='pdf',bbox_inches='tight')
#            print('') 
#            plt.xlabel('#number of cores')
#            plt.ylabel('MFLop/s')
#            plt.title(date_str+'\n\n\nvector size: '+str(s))
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        
#        i=i+2    
#plt.show()
#pp.close()
#
#plt.figure(i)
#for s in sizes:
#    for t in thr:
#        a=[]
#        b=[]
#    #    plt.figure(t+1)
#        for k in d_all[s].keys():
#            if 'openmp' not in k:
#                a.append(d_all[s][k][t])
#                b.append(int(k.split('-')[0]))
#                
#        plt.scatter(b,a, label=str(t+1)+" threads")
#        plt.xlabel('chunk size')
#        plt.ylabel('MFLop/s')
#        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        plt.xscale('log')
#        plt.grid(True, 'both')
#        plt.title(date_str)
#    
#        
#    plt.savefig(pp, format='pdf',bbox_inches='tight')
#    print('')
    

#    
#   
#i=1 
#for s in sizes[4:-2]:
#    t=15
#    a=[]
#    b=[]
#    plt.figure(i)
#    for k in d_all[s].keys():            
#        a.append(d_all[s][k][t])
#        b.append(int(k))
##    indices=np.argsort(b)  
#    plt.plot(b, a, label=str(s))
#
##    plt.plot([b[i] for i in indices],[a[i] for i in indices], label=str(s))
#    plt.xlabel('chunk size')
#    plt.xscale('log')
#    plt.ylabel('MFLop/s')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    i=i+1
#    
#    
#    
#
#
#benchmark='dvecdvecadd'
#chunk_size=1
#th=16
#
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#L=[('r', 'o'),('b', '^')]
#i=0
#
#for s in sizes:
#    xs=[]
#    ys=[]
#    zs=[]
##    for th in thr:
#    for cs in d_all[benchmark][s].keys():
#        if cs!=0:
#            xs.append(np.log10(s))
#            zs.append(d_all[benchmark][s][cs][th-1])
#    #        zs.append(th)
#            ys.append(np.log10(cs))
#    xs=np.asarray(xs)
#    ys=np.asarray(ys)
#    zs=np.asarray(zs)
#    ax.scatter(xs, ys, zs)
#    i=i+1
#ax.set_yticks(np.log10(ds))
#ax.set_xticks(np.log10([100,1000,10000,100000,1000000,10000000]))
#
#ax.set_xlabel('vector size')
#ax.set_zlabel('MFlops')
#ax.set_ylabel('chunk_sizes')
#
##ax.set_zlabel('#cores')
#
#
#
#
#benchmark='dvecdvecadd'
#chunk_size=1
#th=16
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#L=[('r', 'o'),('b', '^')]
#i=0
#for s in sizes:
#    for cs in d_all[benchmark][s].keys():
#        xs=[]
#        ys=[]
#        zs=[]
#        for th in thr:
#            xs.append(s)
#            zs.append(d_all[benchmark][s][cs][th-1])
#            ys.append(th)
#        xs=np.asarray(np.log10(xs))
#        ys=np.asarray(ys)
#        zs=np.asarray(zs)
#        ax.scatter(xs, ys, zs)
#        i=i+1
#ax.set_xticks(np.log10(sizes))
#ax.set_xlabel('vector size')
#ax.set_zlabel('MFlops')
#ax.set_ylabel('#cores')
#
#
