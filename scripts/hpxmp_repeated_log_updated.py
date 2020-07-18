#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:37:23 2018

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")
benchmark='dvecdvecadd'
#benchmark='daxpy'
#benchmark='dmatdmatadd'
#benchmark='dmatdmatmult'
#openmp_date_str='12-14-2018-1512'
#openmp_date_str='01-17-2019-1242'
#openmp_date_str='11-19-18-0936' #daxpy
#openmp_date_str='12-14-2018-1512' #dmatdmatmult
openmp_date_str='01-17-2019-1242' #dmatdmatadd

openmp_dir_1='/home/shahrzad/repos/Blazemark/data/openmp/all/'
openmp_dir_2='/home/shahrzad/repos/Blazemark/data/openmp/04-27-2019/'

hpxmp_dir_1='/home/shahrzad/repos/Blazemark/data/hpxmp/all'
hpxmp_dir_2='/home/shahrzad/repos/Blazemark/data/hpxmp/2/idle_on'
hpxmp_dir_3='/home/shahrzad/repos/Blazemark/data/hpxmp/4'
hpx_ref_date_str='11-22-18-1027'  #reference hpx dvecdvecadd
hpx_ref_dir='/home/shahrzad/repos/Blazemark/data/vector/'+hpx_ref_date_str
hpx_counters_dir='/home/shahrzad/repos/Blazemark/data/hpxmp/3/dvecdvecadd_6repeat_201216size/hpx'
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/hpxmp/4'

from matplotlib.backends.backend_pdf import PdfPages

def create_dict(directory):
    thr=[]
    repeats=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmark=''
    benchmarks=[]
    for filename in data_files:
        (repeat, benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')         
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)                
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))          
                  
    thr.sort()
    benchmarks.sort()      
    repeats.sort()      
    
    d_all={}   
    d={}
    for benchmark in benchmarks:  
        d_all[benchmark]={}
        d[benchmark]={}
        for th in thr:
            d_all[benchmark][th]={}
            d[benchmark][th]={}
            for r in repeats:
                d_all[benchmark][th][r]={}        
                                            
    data_files.sort()        
    for filename in data_files:                
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])        
        size=[]
        mflops=[]    
        for r in result:        
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
            
        d_all[benchmark][th][repeat]['size']=size
        d_all[benchmark][th][repeat]['mflops']=mflops
 
        
    for benchmark in benchmarks:
        for th in thr:
            d[benchmark][th]['size']=d_all[benchmark][th][1]['size']
            mflops=[0]*len(d[benchmark][th]['size'])    
            if max(repeats)==1:
                if 'mflops' in d_all[benchmark][th][1].keys():
                    mflops=d_all[benchmark][th][repeats[0]]['mflops']
                    d[benchmark][th]['mflops']=mflops
            else:
                for r in repeats[1:]:
                    mflops=[mflops[i]+d_all[benchmark][th][r]['mflops'][i] for i in range(len(mflops))]                        
                    d[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
    return d                           
           
#####################################################
                        #hpx refernce
#####################################################
d_hpx_ref=create_dict(hpx_ref_dir)                                              

####################################################
#openmp
####################################################   
d_openmp_1=create_dict(openmp_dir_1)                        
d_openmp_2=create_dict(openmp_dir_2)                        

####################################################
#hpxmp_1
####################################################        
d_hpxmp_1=create_dict(hpxmp_dir_1)
        
####################################################
#hpxmp_2
####################################################    
d_hpxmp_2=create_dict(hpxmp_dir_2)
    
####################################################
#hpxmp_3
####################################################    
d_hpxmp_3=create_dict(hpxmp_dir_3)

####################################################
#hpx performance counters
####################################################        
data_files=glob.glob(hpx_counters_dir+'/*.dat')
d_hpxmp_counters={}

thr=[]
repeats=[]
hpxmp_counters_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpxmp_counters_benchmarks:
            hpxmp_counters_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
hpxmp_counters_benchmarks.sort()      
repeats.sort()      
d_hpxmp_counters_all={}   

for repeat in repeats:
    d_hpxmp_counters_all[repeat]={}
    for benchmark in hpxmp_counters_benchmarks:   
        d_hpxmp_counters_all[repeat][benchmark]={}
        for th in thr:
            d_hpxmp_counters_all[repeat][benchmark][th]={}


data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    
    if 'hpxmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        if th in thr:
            counters={'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th}

            for r in result:
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
                elif "threads" in r:
                    if 'idle-rate' in r and 'pool' in r:
                        idle_rate=float(r.strip().split(',')[-2])/100
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['idle_rate'][th_num]=idle_rate
                    if 'average,' in r and 'pool' in r:
                        average_time=float(r.strip().split(',')[-2])/1000
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['average_time'][th_num]=average_time
                    if 'cumulative,' in r and 'pool' in r:
                        cumulative=float(r.strip().split(',')[-1])
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['cumulative_count'][th_num]=cumulative
                    if 'cumulative-overhead' in r and 'pool' in r:
                        cumulative_overhead=float(r.strip().split(',')[-2])/1000
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['cumulative_overhead_time'][th_num]=cumulative_overhead
                    if 'average-overhead' in r and 'pool' in r:
                        average_overhead=float(r.strip().split(',')[-2])/1000
                        th_num=int(r.strip().split('thread#')[1].split('}')[0])
                        counters['average_overhead_time'][th_num]=average_overhead                        
            d_hpxmp_counters_all[repeat][benchmark][th]['size']=size
            d_hpxmp_counters_all[repeat][benchmark][th]['mflops']=mflops
            d_hpxmp_counters_all[repeat][benchmark][th]['counters']=counters
            
d_hpxmp_counters={}
for benchmark in hpxmp_counters_benchmarks:
    d_hpxmp_counters[benchmark]={}
    for th in thr:        
        d_hpxmp_counters[benchmark][th]={}
        size_0=d_hpxmp_counters_all[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        counter_d_0=[0]*th
        counter_d_1=[0]*th
        counter_d_2=[0]*th
        counter_d_3=[0]*th
        counter_d_4=[0]*th
        d_hpxmp_counters[benchmark][th]['counters']={}
        for r in repeats[1:]:        
            size=d_hpxmp_counters_all[r][benchmark][th]['size']
            mflops=[mflops[i]+d_hpxmp_counters_all[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]                
            counter_d_0=[counter_d_0[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['idle_rate'][i] for i in range(th)]
            counter_d_1=[counter_d_1[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['average_time'][i] for i in range(th)]
            counter_d_2=[counter_d_2[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['cumulative_overhead_time'][i] for i in range(th)]
            counter_d_3=[counter_d_3[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['cumulative_count'][i] for i in range(th)]
            counter_d_4=[counter_d_4[i]+d_hpxmp_counters_all[r][benchmark][th]['counters']['average_overhead_time'][i] for i in range(th)]

            
        d_hpxmp_counters[benchmark][th]['size']=size_0
        d_hpxmp_counters[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
        d_hpxmp_counters[benchmark][th]['counters']['idle_rate']=[x/float(max(repeats)-1) for x in counter_d_0]
        d_hpxmp_counters[benchmark][th]['counters']['average_time']=[x/float(max(repeats)-1) for x in counter_d_1]
        d_hpxmp_counters[benchmark][th]['counters']['cumulative_overhead_time']=[x/float(max(repeats)-1) for x in counter_d_2]
        d_hpxmp_counters[benchmark][th]['counters']['cumulative_count']=[x/float(max(repeats)-1) for x in counter_d_3]
        d_hpxmp_counters[benchmark][th]['counters']['average_overhead_time']=[x/float(max(repeats)-1) for x in counter_d_4]
i=1
for benchmark in hpxmp_counters_benchmarks:  
    plt.figure(i)
    j=d_openmp[benchmark][th]['size'].index(size[0])
    pp = PdfPages(perf_directory+'/hpx_performance.pdf')
    plt.plot(thr, [d_openmp[benchmark][th]['mflops'][j] for th in thr], label='openmp')
    plt.plot(thr, [d_hpxmp_counters[benchmark][th]['mflops'][0] for th in thr], label='hpx')
    plt.xlabel("#threads")           
    plt.ylabel('MFlops')
    plt.grid(True, 'both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(benchmark+ " vector size 201,216")
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    plt.show()
    pp.close() 
    i=i+1
    j=d_openmp[benchmark][th]['size'].index(size[0])
    for th in thr:
        pp = PdfPages(perf_directory+'/hpx_performance_counters_'+str(th)+'.pdf')

        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['idle_rate'], label='idle_rate')
        plt.xlabel("#threads")      
        plt.xticks(np.arange(1,th+1).tolist())
        plt.ylabel('%')
        plt.title('idle_rate')
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        i=i+1
        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['average_time'], label='average_time')
        plt.xlabel("#threads")     
        plt.xticks(np.arange(1,th+1).tolist())
        plt.ylabel('Microseconds')
        plt.title('average_time')
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        i=i+1
        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['cumulative_overhead_time'], label='cumulative_overhead_time')
        plt.xlabel("#threads") 
        plt.xticks(np.arange(1,th+1).tolist())                   
        plt.ylabel('Microseconds')
        plt.title('cumulative_overhead_time')
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        i=i+1
        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['cumulative_count'], label='cumulative_count')
        plt.xlabel("#threads") 
        plt.xticks(np.arange(1,th+1).tolist())                   
#        plt.ylabel('')
        plt.title("cumulative_count")
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        i=i+1
        plt.figure(i)
        plt.bar(np.arange(1,th+1).tolist(), d_hpxmp_counters[benchmark][th]['counters']['average_overhead_time'], label='average_overhead_time')
        plt.xlabel("#threads")    
        plt.xticks(np.arange(1,th+1).tolist())                   
        plt.ylabel('Microseconds')
        plt.title("average_overhead_time")
        i=i+1
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
        plt.show()
        pp.close() 
    
    
        
##############################################################################
        #plots
##############################################################################
#pp = PdfPages(perf_directory+'/hpxmp_idle.pdf')
benchmarks=['dvecdvecadd', 'dmatdmatadd']        
i=1
for benchmark in benchmarks:       
    for th in [1, 4,8,16]:
#        pp = PdfPages(perf_directory+'/figures/'+benchmark+'_'+str(th)+'.pdf')

        plt.figure(i)

        plt.plot(d_hpxmp_1[benchmark][th]['size'], d_hpxmp_1[benchmark][th]['mflops'],label="hpxMP previous",color='black',linestyle=':')
#        if benchmark in d_hpxmp_2.keys():
#            plt.plot(d_hpxmp_2[benchmark][th]['size'], d_hpxmp_2[benchmark][th]['mflops'],label='hpxmp_idle_on '+str(th)+' threads')
        plt.plot(d_openmp_2[benchmark][th]['size'], d_openmp_2[benchmark][th]['mflops'],label="llvm-OpenMP", color='black')
        plt.plot(d_hpxmp_3[benchmark][th]['size'], d_hpxmp_3[benchmark][th]['mflops'],label="hpxMP", color='black', linestyle='dashed')


#        plt.plot(d_hpxmp_3[benchmark][th]['size'], d_hpxmp_3[benchmark][th]['mflops'],label="hpxmp "+str(th)+" threads", color='black', linestyle='dashed')
#        plt.plot(d_hpx[benchmark][10][256][th]['size'], d_hpx[benchmark][10][256][th]['mflops'],label='hpx '+str(th)+' threads')

        plt.xlabel("size $n$")           
        plt.ylabel('MFlops ('+str(th)+" threads)")
        plt.xscale('log')
        plt.grid(True, 'both')
        if i==5:
            plt.legend(loc=1)
        else:   
            plt.legend(loc=2)
#        plt.title(benchmark)
        i=i+1
#        plt.savefig(pp, format='pdf',bbox_inches='tight')
#        plt.show()
#        pp.close() 
#    print('')
#plt.show()
#pp.close() 
        
#for th in thr:
#    a=(d_hpxmp_2['dvecdvecadd'][th]['mflops'][135]-d_hpxmp_1['dvecdvecadd'][th]['mflops'][135])/d_hpxmp_1['dvecdvecadd'][th]['mflops'][135]
#    print(str(th)+': '+str(a))

#th=16     
#s=0   
#for i in range(1,12):
#    print(d_hpxmp_all_1[i]['dvecdvecadd'][th]['mflops'][45])
#    print(d_hpxmp_all_2[i]['dvecdvecadd'][th]['mflops'][45])
#    print(d_hpxmp_all_3[i]['dvecdvecadd'][th]['mflops'][45])
#
#    print('----')
#    
#for th in thr:    
#    s=[0]*len(d_hpxmp_all_1[1]['dvecdvecadd'][th]['mflops'])
#    for r in repeats[1:]:
#        s= [s[i]+d_hpxmp_all_1[r]['dvecdvecadd'][th]['mflops'][i] for i in range(len(mflops))]
#    s=[s[i]/(len(repeats)-1) for i in range(len(mflops))]
#    v=[0]*len(d_hpxmp_all_1[1]['dvecdvecadd'][th]['mflops'])
#    for r in repeats[1:]:
#        v= [v[i]+(d_hpxmp_all_1[r]['dvecdvecadd'][th]['mflops'][i]-s[i])*(d_hpxmp_all_1[r]['dvecdvecadd'][th]['mflops'][i]-s[i]) for i in range(len(mflops))] 
#    v=[np.sqrt(v[i])/((len(repeats)-1)*s[i]) for i in range(len(mflops))]
#    
N = [1, 2, 4, 8, 16]
benchmark='dvecdvecadd'
grain_sizes=d_openmp_2[benchmark][16]['size']

Xs=[[d_openmp_2[benchmark][n]['mflops'][g] for n in N] for g in range(len(grain_sizes))]

models = [ usl.usl(N, X) for X in Ts]
plt.plot(grain_sizes, Ts)

import scipy.interpolate as interp
import scipy.optimize as optimize
rs = np.linspace(0, grain_sizes[-1], 10000)

grain_model = []

#plt.subplot(323)
for idx in range(len(N)):
    plt.figure(2 + idx)
    plt.title('Varying Grain Sizes, N=%s' % N[idx])
    X = [ x[idx] for x in Xs ]
    X = np.array(X)
    X_mean = X.mean()
    X = X / X_mean
    G = np.array(grain_sizes)
    G_mean = G.mean()
    G = G / G_mean
    plt.plot(grain_sizes, X * X_mean, 'ok', label='Original Data, N = %s' % N[idx])
    plt.plot(grain_sizes, X * X_mean, 'k--')

    [popt, pcov] = curve_fit(rational, G, X, bounds=(0, np.inf))
    plt.plot(rs, rational(rs / G_mean, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) * X_mean, label='fitted, n = %s, P(x)/Q(x)' % (N[idx]))

    [popt, pcov] = curve_fit(levy, G, X, bounds=(0, np.inf))
    plt.plot(rs[1:], levy(rs[1:] / G_mean, popt[0], popt[1], popt[2]) * X_mean, label='Fitted, n = %s, levy distribution' % (N[idx]))

    plt.legend()

    grain_model.append

plt.show()
