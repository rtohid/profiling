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

now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")

hpx_dir='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/decision_tree/performances/marvin/'
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/predictions/'
hpx_ref_dir_static='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/reference/static/'
hpx_ref_dir_dynamic='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/reference/dynamic/'
hpx_ref_dir_gabriel='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/reference/Gabriel/'
hpx_ref_dir_gabriel_training='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/predictions/1_times_thread/'
hpx_ref_dir_gabriel_chunks_cores='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/equal_chunks/'
hpx_ref_dir_gabriel_1='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/reference/without4'
hpx_ref_dir_gabriel_8='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/reference/with8'
hpx_ref_dir_gabriel_5='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/reference/with5'

blocks_dir_1='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/block_sizes/1_times_thread'
blocks_dir_4='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/block_sizes/4_times_thread'
blocks_dir_5='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/block_sizes/5_times_thread'
blocks_dir_8='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/block_sizes/8_times_thread'

equal_chunks_dir='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/chunk_sizes/equal_chunks'
training_chunks_dir='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/chunk_sizes/training'
full_chunks_dir='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/chunk_sizes/fullset'

perf_counters='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/Gabriel/performance_counters/each_size'

from matplotlib.backends.backend_pdf import PdfPages

def read_chunks(directory):
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
    d={}
    for node in nodes:
        d_all[node]={}
        d[node]={}
        for benchmark in benchmarks:  
            d_all[node][benchmark]={}
            d[node][benchmark]={}
            for th in thr:
                d_all[node][benchmark][th]={}     
                d[node][benchmark][th]=[]

                                            
    data_files.sort()       
    
    for filename in data_files:                
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        try:
            (node, benchmark, th) = filename.split('/')[-1].replace('.dat','').split('-')   
        except:
            (node, benchmark, th, ref) = filename.split('/')[-1].replace('.dat','').split('-')          
        th = int(th)
   
        for r in result:        
            if "matrix" in r:
                m_size=int(r.strip().split('matrix size: ')[1].split(' block')[0])
                b_size=(r.strip().split('block size: ')[1].split(' num')[0])
                num_iter=int(r.strip().split('iterations: ')[1].split(' chunk')[0])
                chunk_size=int(r.strip().split('chunk size: ')[1].split('\n')[0])
                if (m_size, b_size, num_iter, chunk_size) not in d[node][benchmark][th]:
                    d[node][benchmark][th].append((m_size, b_size, num_iter, chunk_size))
                if m_size not in d_all[node][benchmark][th].keys():
                    d_all[node][benchmark][th][m_size]={}
                if b_size not in d_all[node][benchmark][th][m_size].keys():
                    d_all[node][benchmark][th][m_size][b_size]={}
                if num_iter not in d_all[node][benchmark][th][m_size][b_size].keys():
                    d_all[node][benchmark][th][m_size][b_size]={}
                d_all[node][benchmark][th][m_size][b_size][num_iter]=chunk_size
            
    return d                          

def read_blocks(directory):
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
    
    d={}
    for node in nodes:
        d[node]={}
        for benchmark in benchmarks:  
            d[node][benchmark]={}
            for th in thr:
                d[node][benchmark][th]={}

                                            
    data_files.sort()       
    
    for filename in data_files:                
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        try:
            (node, benchmark, th) = filename.split('/')[-1].replace('.dat','').split('-')   
        except:
            (node, benchmark, th, ref) = filename.split('/')[-1].replace('.dat','').split('-')          
        th = int(th)
        for r in result:        
            if "," in r:
                (m_size, m, n, block_row, block_col)=[int(i) for i in r.split(',')]                          
#                if m_size not in d[node][benchmark][th]:
                mflop=0
                if 'add' in benchmark:                    
                    mflop=block_row*block_col
                else:
                    mflop=block_row*(2*m_size*m_size)
                
                rest1=block_row%simdsize
                rest2=block_col%simdsize      
                if rest1!=0:
                    block_row+=simdsize-rest1
                if rest2!=0:
                    block_col+=simdsize-rest2                
                d[node][benchmark][th][m_size]=( m, n, block_row, block_col, mflop)  

    return d              

def create_dict(directory):
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
           
#####################################################
                        #hpx refernce
#####################################################
d_hpx=create_dict(hpx_dir)                                              

d_hpx_ref_static=create_dict(hpx_ref_dir_static)                                              

d_hpx_ref_dynamic=create_dict(hpx_ref_dir_dynamic)                                              
d_hpx_ref_gabriel=create_dict(hpx_ref_dir_gabriel)                                              
d_hpx_ref_gabriel_training=create_dict(hpx_ref_dir_gabriel_training)                                              
d_hpx_ref_gabriel_chunks_cores=create_dict(hpx_ref_dir_gabriel_chunks_cores)                                              
d_hpx_ref_gabriel_1=create_dict(hpx_ref_dir_gabriel_1)                                              
d_hpx_ref_gabriel_8=create_dict(hpx_ref_dir_gabriel_8)                                              
d_hpx_ref_gabriel_5=create_dict(hpx_ref_dir_gabriel_5)                                              


##############################################################################
        #plots
##############################################################################
equal_chunks=read_chunks(equal_chunks_dir)
training_chunks=read_chunks(training_chunks_dir)
full_chunks=read_chunks(full_chunks_dir)
    
fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))
#pp = PdfPages(perf_directory+'/recent/old_backendvs_ML.pdf')
i=1
for node in d_hpx.keys():
    for benchmark in d_hpx[node].keys():     
#        pp = PdfPages(perf_directory +benchmark+'.pdf')

        for th in d_hpx[node][benchmark].keys():
    
            plt.figure(i)
            fig, ax = plt.subplots()
#            plt.plot(d_hpx[node][benchmark][th]['size'], d_hpx[node][benchmark][th]['mflops'],label="ML full tree "+str(th)+" threads")
#            plt.plot(d_hpx_ref_static[node][benchmark][th]['size'], d_hpx_ref_static[node][benchmark][th]['mflops'],label="old_backend static"+str(th)+" threads")
#            plt.plot(d_hpx_ref_gabriel[node][benchmark][th]['size'], d_hpx_ref_gabriel[node][benchmark][th]['mflops'],label="gabriel's branch ref "+str(th)+" threads")
            ax.plot(d_hpx_ref_gabriel_training[node][benchmark][th]['size'], d_hpx_ref_gabriel_training[node][benchmark][th]['mflops'],label="ML only training "+str(th)+" threads")
#            plt.plot(d_hpx_ref_gabriel_chunks_cores[node][benchmark][th]['size'], d_hpx_ref_gabriel_chunks_cores[node][benchmark][th]['mflops'],label="#chunks=niter/cores "+str(th)+" threads")
#            plt.plot(d_hpx_ref_gabriel_1[node][benchmark][th]['size'], d_hpx_ref_gabriel_1[node][benchmark][th]['mflops'],label="old_backend 1*num_threads "+str(th)+" threads")
            ax.plot(d_hpx_ref_dynamic[node][benchmark][th]['size'], d_hpx_ref_dynamic[node][benchmark][th]['mflops'],label="old_backend 4*num_threads"+str(th)+" threads")
#            plt.plot(d_hpx_ref_gabriel_5[node][benchmark][th]['size'], d_hpx_ref_gabriel_5[node][benchmark][th]['mflops'],label="old_backend 5*num_threads "+str(th)+" threads")
#            plt.plot(d_hpx_ref_gabriel_8[node][benchmark][th]['size'], d_hpx_ref_gabriel_8[node][benchmark][th]['mflops'],label="old_backend 8*num_threads "+str(th)+" threads")
#
            for i in len(d_hpx_ref_dynamic[node][benchmark][th]['size']):
                ax.annotate(training_chunks[node][benchmark][th], (z[i], y[i]))
            plt.title(benchmark)
            plt.xlabel("# matrix size")           
            plt.ylabel('MFlops')
            plt.xscale('log')
            plt.grid(True, 'both')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            i=i+1
#            plt.savefig(pp, format='pdf',bbox_inches='tight')
#            print('')
#plt.show()
#pp.close() 


###############################################################################
#perforance counters
###############################################################################
def create_dict_counters(directory):
    thr=[]
    nodes=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmarks=[]
    mults=[]
    
    for filename in data_files:
        (node, benchmark, th, runtime, mult, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)                
        if int(th) not in thr:
            thr.append(int(th))
        if node not in nodes:
            nodes.append(node)   
        if int(mult) not in mults:
            mults.append(int(mult))
                  
    thr.sort()
    nodes.sort()
    benchmarks.sort()   
    mults.sort()
    repeats=6
    
    d_all={}   
    for node in nodes:
        d_all[node]={}
        for benchmark in benchmarks:  
            d_all[node][benchmark]={}
            for th in thr:
                d_all[node][benchmark][th]={}    
                for mult in mults:
                    d_all[node][benchmark][th][mult]={} 
                                            
    data_files.sort()        
    for filename in data_files:                       
        f=open(filename, 'r')                
        results=f.read()
        try:
            (node, benchmark, th, runtime, mult, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')   
        except:
            print("error reading file name")          
        th = int(th)
        mat_size=int(mat_size)
        mult=int(mult)
        counters_avg={'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th}

        if mat_size not in d_all[node][benchmark][th][mult].keys():
            d_all[node][benchmark][th][mult][mat_size]={'ind':[{}]*repeats, 'avg':{}}


        reps=results.split('First Done')[1:]
        for rep in reps[1:]:
            counters_ind={'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th}            

            rep_lines=rep.split('Finished Initialization')[0].split('\n')   
            for r in rep_lines:
                if 'idle-rate' in r and 'pool' in r:
                    idle_rate=float(r.strip().split(',')[-2])/100
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['idle_rate'][th_num]=idle_rate
                    counters_avg['idle_rate'][th_num]+=idle_rate/repeats
                elif 'cumulative-overhead' in r and 'pool' in r:
                    cumulative_overhead=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['cumulative_overhead_time'][th_num]=cumulative_overhead
                    counters_avg['cumulative_overhead_time'][th_num]+=cumulative_overhead/repeats
                elif 'average-overhead' in r and 'pool' in r:
                    average_overhead=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['average_overhead_time'][th_num]=average_overhead   
                    counters_avg['average_overhead_time'][th_num]+=average_overhead     
                elif 'average,' in r and 'pool' in r:
                    average_time=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['average_time'][th_num]=average_time
                    counters_avg['average_time'][th_num]+=average_time/repeats

                elif 'cumulative,' in r and 'pool' in r:
                    cumulative=float(r.strip().split(',')[-1])
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['cumulative_count'][th_num]=cumulative
                    counters_avg['cumulative_count'][th_num]+=cumulative/repeats

                
            d_all[node][benchmark][th][mult][mat_size]['ind'][reps.index(rep)]=counters_ind
        d_all[node][benchmark][th][mult][mat_size]['avg']=counters_avg

                                                   
    return d_all  

###############################################################################       
#hpx_backend
###############################################################################       

blocks_1=read_blocks(blocks_dir_1)
blocks_4=read_blocks(blocks_dir_4)
blocks_5=read_blocks(blocks_dir_5)
blocks_8=read_blocks(blocks_dir_8)
i=1
for node in d_hpx.keys():
    for benchmark in d_hpx[node].keys():  
        if 'vec' not in benchmark:
            for th in d_hpx[node][benchmark].keys():
                all_mflop_1=[]
                all_mflop_4=[]
                all_mflop_5=[]
                all_mflop_8=[]
                all_sizes=[]
                for m_size in blocks_1['trillian'][benchmark][th].keys():
                    all_sizes.append(m_size)
                    all_mflop_1.append(blocks_1['trillian'][benchmark][th][m_size][-1])
                    all_mflop_4.append(blocks_4['trillian'][benchmark][th][m_size][-1])
                    all_mflop_5.append(blocks_5['trillian'][benchmark][th][m_size][-1])
                    all_mflop_8.append(blocks_8['trillian'][benchmark][th][m_size][-1])
    
                    
                plt.figure(i)
                plt.plot(all_sizes, all_mflop_1,label="old_backend 1*num_threads "+str(th)+" threads")
                plt.plot(all_sizes, all_mflop_4,label="old_backend 4*num_threads"+str(th)+" threads")
                plt.plot(all_sizes, all_mflop_5,label="old_backend 5*num_threads "+str(th)+" threads")
                plt.plot(all_sizes, all_mflop_8,label="old_backend 8*num_threads"+str(th)+" threads")
                plt.title(benchmark)
                plt.xlabel("# matrix size")           
                plt.ylabel('MFlop per thread')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
                i=i+1


###############################################################################       
#performance counters graphs
###############################################################################  
perf=create_dict_counters(perf_counters)     
benchmark='dmatdmatadd'
node='marvin'
th=1           


#plot counter based on nuber of threads
mults=[1,4,5,8]            
i=1
for mat_size in [523, 600, 690, 793, 912]:    
    for mult in mults:
        plt.figure(i)
        stats=perf[node][benchmark][th][mult][mat_size]['ind'][1]
        plt.plot(np.arange(1,th+1), [100*j for j in stats['idle_rate']],label='mult='+str(mult),marker='*')
        plt.xlabel("threads")           
        plt.ylabel('idle_rate')
        plt.title(str(mat_size))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.figure(i+1)
        plt.plot(np.arange(1,th+1), stats['cumulative_count'],label='mult='+str(mult),marker='*')
        plt.xlabel("threads")           
        plt.ylabel('cumulative_count')
        plt.title(str(mat_size))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+2   
                
                
#plot counter based on matrix_sizes for a specific multiplier  
mat_sizes=[523, 600, 690, 793, 912]              
i=1

for mult in mults:
    idle_rates=[]
    cum_counts=[]
    for mat_size in mat_sizes:    
        stats=perf[node][benchmark][th][mult][mat_size]['ind'][1]
        idle_rates.append(100*stats['idle_rate'][0])
        cum_counts.append(stats['cumulative_count'][0])
    plt.figure(i)

    plt.plot(mat_sizes, idle_rates)
    plt.xlabel("matrix size")           
    plt.ylabel('idle_rate')
    plt.title('multiplier '+str(mult))
    plt.figure(i+1)

    plt.plot(mat_sizes, cum_counts)
    plt.xlabel("matrix size")           
    plt.ylabel('cumulative_count')
    plt.title('multiplier '+str(mult))

    i=i+2
        
        
        plt.plot(np.arange(1,th+1), [100*j for j in stats['idle_rate']],label='mult='+str(mult),marker='*')
        plt.xlabel("threads")           
        plt.ylabel('idle_rate')
        plt.title(str(mat_size))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.figure(i+1)
        plt.plot(np.arange(1,th+1), stats['cumulative_count'],label='mult='+str(mult),marker='*')
        plt.xlabel("threads")           
        plt.ylabel('cumulative_count')
        plt.title(str(mat_size))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+2                 
                
                
                
###############################################################################       
#chunk size graphs
###############################################################################            
equal_chunks=read_chunks(equal_chunks_dir)
training_chunks=read_chunks(training_chunks_dir)
full_chunks=read_chunks(full_chunks_dir)


i=1
for node in equal_chunks.keys():
    for benchmark in equal_chunks[node].keys():     
#        pp = PdfPages(perf_directory +benchmark+'.pdf')
        if 'dvec' not in benchmark:
            for th in equal_chunks[node][benchmark].keys():
                matrix_sizes=[m[0] for m in equal_chunks[node][benchmark][th]]
                chunk_sizes_eq=[m[-1] for m in equal_chunks[node][benchmark][th]]
                chunk_sizes_tr=[m[-1] for m in training_chunks[node][benchmark][th]]
                chunk_sizes_full=[m[-1] for m in full_chunks[node][benchmark][th]]
    
                plt.figure(i)
                width=50
                fig, ax = plt.subplots()
                x=np.asarray(matrix_sizes)
                rects1 = ax.bar(x - width/2, chunk_sizes_full, width, label='Full '+str(th)+' threads')
                rects2 = ax.bar(x + width/2, chunk_sizes_tr, width, label='Training '+str(th)+' threads')
                plt.title(benchmark)
                plt.xlabel("# matrix size")           
                plt.ylabel('chunk size')
                plt.xscale('log')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                autolabel(rects1)
                autolabel(rects2)
                
                fig.tight_layout()
                
                plt.show()
                
                
                plt.bar(matrix_sizes, chunk_sizes_eq, [width/m for m in matrix_sizes],label="equal "+str(th)+" threads")
                plt.bar(matrix_sizes, chunk_sizes_tr, width,label="training "+str(th)+" threads")
                plt.bar(matrix_sizes, chunk_sizes_full, width,label="full "+str(th)+" threads")
    
                plt.title(benchmark)
                plt.xlabel("# matrix size")           
                plt.ylabel('chunk size')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
                i=i+1




