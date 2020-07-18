import glob
import numpy as np
from matplotlib import pyplot as plt

def create_dict_relative_norepeat_counters_onebyone(directory):
    thr=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmark=''
    benchmarks=[]
    chunk_sizes=[]
    block_sizes={}
    mat_sizes={}
    nodes=[]
    
    for filename in data_files:
        (node, benchmark, th, runtime, chunk_size, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
        mat_size=mat_size.split(',')[0]
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)   
                mat_sizes[benchmark]=[]
                block_sizes[benchmark]=[]
        if int(mat_size) not in mat_sizes[benchmark]:
            mat_sizes[benchmark].append(int(mat_size))
        if int(th) not in thr:
            thr.append(int(th))        
        if block_size_row+'-'+block_size_col not in block_sizes[benchmark]:
            block_sizes[benchmark].append(block_size_row+'-'+block_size_col)
        if int(chunk_size) not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))
        if node not in nodes:
            nodes.append(node)
                  
    thr.sort()
    nodes.sort()      
    chunk_sizes.sort()
    benchmarks.sort()   
    repeats=5
    
    d={}
    for node in nodes:
        d[node]={}
        for benchmark in benchmarks:  
            mat_sizes[benchmark].sort()
            block_sizes[benchmark].sort()
            d[node][benchmark]={}
           
            for th in thr:
                d[node][benchmark][th]={}
                for bs in block_sizes[benchmark]:
                    d[node][benchmark][th][bs]={}
                    for cs in chunk_sizes:
                        d[node][benchmark][th][bs][cs]={}
                        d[node][benchmark][th][bs][cs]['size']=mat_sizes[benchmark]
                        d[node][benchmark][th][bs][cs]['mflops']=[0]*len(mat_sizes[benchmark])
                        d[node][benchmark][th][bs][cs]['counters']=[0]*len(mat_sizes[benchmark])

    data_files.sort()        
    for filename in data_files:                
        f=open(filename, 'r')
                 
        results=f.read()
        (node, benchmark, th, runtime, chunk_size, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
        th=int(th)       
        cs=int(chunk_size)     
        counters_avg={'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th, 'papi_tca':[0]*th, 'papi_tcm':[0]*th}
        s=mat_sizes[benchmark].index(int(mat_size))

        bs=block_size_row+'-'+block_size_col
        mflops=float((results.split(' '+mat_size+' ')[1].split('\n')[0]).strip())
        d[node][benchmark][th][bs][cs]['mflops'][s]=mflops
        s=mat_sizes[benchmark].index(int(mat_size))
        d[node][benchmark][th][bs][cs]['counters'][s]={}
        d[node][benchmark][th][bs][cs]['counters'][s]['ind']=[]
        d[node][benchmark][th][bs][cs]['counters'][s]['avg']={}
        
        reps=results.split('Done')[1:]
        for rep in reps[1:-1]:
            counters_ind={'idle_rate':[0]*th, 'average_time':[0]*th, 'cumulative_overhead_time':[0]*th, 'cumulative_count':[0]*th, 'average_overhead_time':[0]*th,'papi_tca':[0]*th, 'papi_tcm':[0]*th}            

            rep_lines=rep.split('Initialization')[0].split('\n')   
            for r in rep_lines:
                if 'idle-rate' in r and 'pool' in r:
                    idle_rate=float(r.strip().split(',')[-2])/100
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['idle_rate'][th_num]=idle_rate
                    counters_avg['idle_rate'][th_num]+=idle_rate
                elif 'cumulative-overhead' in r and 'pool' in r:
                    cumulative_overhead=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['cumulative_overhead_time'][th_num]=cumulative_overhead
                    counters_avg['cumulative_overhead_time'][th_num]+=cumulative_overhead
                elif 'average-overhead' in r and 'pool' in r:
                    average_overhead=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['average_overhead_time'][th_num]=average_overhead   
                    counters_avg['average_overhead_time'][th_num]+=average_overhead     
                elif 'average,' in r and 'pool' in r:
                    average_time=float(r.strip().split(',')[-2])/1000
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['average_time'][th_num]=average_time
                    counters_avg['average_time'][th_num]+=average_time
                elif 'cumulative,' in r and 'pool' in r:
                    cumulative=float(r.strip().split(',')[-1])
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['cumulative_count'][th_num]=cumulative
                    counters_avg['cumulative_count'][th_num]+=cumulative
                elif 'PAPI_L2_TCA' in r :
                    papi_tca=float(r.strip().split(',')[-1])
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['papi_tca'][th_num]=papi_tca
                    counters_avg['papi_tca'][th_num]+=papi_tca
                elif 'PAPI_L2_TCM' in r :
                    papi_tca=float(r.strip().split(',')[-1])
                    th_num=int(r.strip().split('thread#')[1].split('}')[0])
                    counters_ind['papi_tcm'][th_num]=papi_tca
                    counters_avg['papi_tcm'][th_num]+=papi_tca

                        
            d[node][benchmark][th][bs][cs]['counters'][s]['ind'].append(counters_ind)
        for counter in counters_avg.keys():
            counters_avg[counter]=[counters_avg[counter][thread]/repeats for thread in range(th)]
        d[node][benchmark][th][bs][cs]['counters'][s]['avg']=counters_avg

    return (d, chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)  




papi_directory='/home/shahrzad/repos/Blazemark/data/matrix/08-07-2019/performance_counters/marvin/'
(d_hpx,  chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)=create_dict_relative_norepeat_counters_onebyone(papi_directory)                 

