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
#benchmark='dvecdvecadd'
#benchmark='daxpy'
benchmark='dmatdmatadd'
#benchmark='dmatdmatmult'
#openmp_date_str='12-14-2018-1512'
#openmp_date_str='01-17-2019-1242'
#openmp_date_str='11-19-18-0936' #daxpy
#openmp_date_str='12-14-2018-1512' #dmatdmatmult
openmp_date_str='01-17-2019-1242' #dmatdmatadd
openmp_dir='/home/shahrzad/repos/Blazemark/data/openmp/'+benchmark+'/'+openmp_date_str


openmp_chunks_date_str='01-17-2019-1242' #dmatdmatadd
openmp_chunks_dir='/home/shahrzad/repos/Blazemark/data/openmp/'+benchmark+'/chunks'

#openmp_dir='/home/shahrzad/repos/Blazemark/data/openmp/all'
#hpx_date_str='01-18-2019-1105'  #dmatdmatadd new hpx
hpx_date_str='01-04-2019-1027' #dmatdmatadd
hpx_date_str = '03-01-2019-hpx_updated_1row'
hpx_dir='/home/shahrzad/repos/Blazemark/data/matrix/'+benchmark+'/'+hpx_date_str

#hpx_date_str='12-20-2018-0936'
#px_before_date_str='12-10-18-0935'  #reference hpx dmatdmatmult
hpx_before_date_str='01-06-2019-1059'  #reference hpx dmatdmatadd
#hpx_date_str='aggregated/01-18-2019-0924'
hpxmp_date_str='01-19-2019-1500'
hpxmp_dir='/home/shahrzad/repos/Blazemark/data/hpxmp/dmatdmatadd/'+hpxmp_date_str

hpxmp_dir='/home/shahrzad/repos/Blazemark/data/hpxmp/all'

perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots'

from matplotlib.backends.backend_pdf import PdfPages

    
#####################################################
                        #hpx refernce
#####################################################
hpx_before_dir='/home/shahrzad/repos/Blazemark/data/matrix/'+benchmark+'/reference/'+hpx_before_date_str                    
thr=[]
sizes=[]
repeats=[]

data_files=glob.glob(hpx_before_dir+'/*.dat')

hpx_before_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:
        (repeat, benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpx_before_benchmarks:
            hpx_before_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
hpx_before_benchmarks.sort()      
repeats.sort()      


d_hpx_before_all={}   
d_hpx_before={}
for benchmark in hpx_before_benchmarks:  
    d_hpx_before_all[benchmark]={}
    d_hpx_before[benchmark]={}    
    for th in thr:
        d_hpx_before_all[benchmark][th]={}
        d_hpx_before[benchmark][th]={}
        for r in repeats:
            d_hpx_before_all[benchmark][th][r]={}
                
data_files.sort()        
for filename in data_files:    
    if 'hpx' in filename:
        size=[]
        mflops=[]    
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
            
        for r in result:        
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
        
        d_hpx_before_all[benchmark][th][repeat]['size']=size
        d_hpx_before_all[benchmark][th][repeat]['mflops']=mflops
 
    
for benchmark in hpx_before_benchmarks:    
    for th in thr:
        mflops=[0]*len(size)
        d_hpx_before[benchmark][th]['size']=size

        if max(repeats)==1:
            mflops=d_hpx_before_all[benchmark][th][repeats[0]]['mflops']
            d_hpx_before[benchmark][th]['mflops']=mflops
        else:
            for r in repeats[1:]:
                mflops=[mflops[i]+d_hpx_before_all[benchmark][th][r]['mflops'][i] for i in range(len(mflops))]                        
                d_hpx_before[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops] 
########################################################################
#hpx
##########################################################################                        
thr=[]
sizes=[]
repeats=[]
block_sizes=[]
chunk_sizes=[]
block_sizes_row=[]
block_sizes_col=[]

data_files=glob.glob(hpx_dir+'/*.dat')

hpx_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:
        if len(filename.split('/')[-1].replace('.dat','').split('-'))==6:
            option=1
            (repeat, benchmark, th, runtime, chunk_size, block_size) = filename.split('/')[-1].replace('.dat','').split('-')
            if benchmark not in hpx_benchmarks:
                hpx_benchmarks.append(benchmark)
                
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat))
            if int(block_size) not in block_sizes:
                block_sizes.append(int(block_size))
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
        elif len(filename.split('/')[-1].replace('.dat','').split('-'))==7:
            option=2
            (repeat, benchmark, th, runtime, chunk_size, block_size_row, block_size_col) = filename.split('/')[-1].replace('.dat','').split('-')
            if benchmark not in hpx_benchmarks:
                hpx_benchmarks.append(benchmark)
                
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat))
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
            if str(block_size_row)+'-'+str(block_size_col) not in block_sizes:
                block_sizes.append(str(block_size_row)+'-'+str(block_size_col))                
        
thr.sort()
hpx_benchmarks.sort()      
repeats.sort()      
block_sizes.sort()
chunk_sizes.sort()

d_hpx_all={}   
d_hpx={}
for benchmark in hpx_benchmarks:  
    d_hpx_all[benchmark]={}
    d_hpx[benchmark]={}
    for c in chunk_sizes:
        d_hpx_all[benchmark][c]={}
        d_hpx[benchmark][c]={}
        for b in block_sizes:
            d_hpx_all[benchmark][c][b]={}
            d_hpx[benchmark][c][b]={}
            for th in thr:
                d_hpx_all[benchmark][c][b][th]={}
                d_hpx[benchmark][c][b][th]={}
                for r in repeats:
                    d_hpx_all[benchmark][c][b][th][r]={}        
                                        
data_files.sort()        
for filename in data_files:    
    if 'hpx' in filename:
        
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        chunk=int(filename.split('/')[-1].split('-')[4]) 
        if option ==1:
            block=int(filename.split('/')[-1].split('-')[5][0:-4])       
        elif option ==2:
            block=filename.split('/')[-1].split('-')[5]+'-'+filename.split('/')[-1].split('-')[6][0:-4]     

        if block in block_sizes:  
            size=[]
            mflops=[]    
            for r in result:        
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
            
            d_hpx_all[benchmark][chunk][block][th][repeat]['size']=size
            d_hpx_all[benchmark][chunk][block][th][repeat]['mflops']=mflops
 
    
for benchmark in hpx_benchmarks:
    for c in chunk_sizes:
        for b in block_sizes:
            for th in thr:
                mflops=[0]*len(size)

                if max(repeats)==1:
                    if 'mflops' in d_hpx_all[benchmark][c][b][th][1].keys():
                        d_hpx[benchmark][c][b][th]['size']=size
                        mflops=d_hpx_all[benchmark][c][b][th][repeats[0]]['mflops']
                        d_hpx[benchmark][c][b][th]['mflops']=mflops
                else:
                    d_hpx[benchmark][c][b][th]['size']=size
                    for r in repeats[1:]:
                        mflops=[mflops[i]+d_hpx_all[benchmark][c][b][th][r]['mflops'][i] for i in range(len(mflops))]                        
                        d_hpx[benchmark][c][b][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
       
        
        
########################################################################
#openmp with chunks
##########################################################################                        
thr=[]
sizes=[]
repeats=[]
block_sizes=[]
chunk_sizes=[]
block_sizes_row=[]
block_sizes_col=[]

data_files=glob.glob(openmp_chunks_dir+'/*.dat')

openmp_chunks_benchmarks=[]
for filename in data_files:
    if 'openmp' in filename:
        if len(filename.split('/')[-1].replace('.dat','').split('-'))==6:
            option=1
            (repeat, benchmark, th, runtime, chunk_size, block_size) = filename.split('/')[-1].replace('.dat','').split('-')
            if benchmark not in openmp_chunks_benchmarks:
                openmp_chunks_benchmarks.append(benchmark)
                
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat))
            if int(block_size) not in block_sizes:
                block_sizes.append(int(block_size))
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
        elif len(filename.split('/')[-1].replace('.dat','').split('-'))==7:
            option=2
            (repeat, benchmark, th, runtime, chunk_size, block_size_row, block_size_col) = filename.split('/')[-1].replace('.dat','').split('-')
            if benchmark not in openmp_chunks_benchmarks:
                openmp_chunks_benchmarks.append(benchmark)
                
            if int(th) not in thr:
                thr.append(int(th))
            if int(repeat) not in repeats:
                repeats.append(int(repeat))
            if int(chunk_size) not in chunk_sizes:
                chunk_sizes.append(int(chunk_size))
            if str(block_size_row)+'-'+str(block_size_col) not in block_sizes:
                block_sizes.append(str(block_size_row)+'-'+str(block_size_col))                
        
thr.sort()
openmp_chunks_benchmarks.sort()      
repeats.sort()      
block_sizes.sort()
chunk_sizes.sort()

d_openmp_chunks_all={}   
d_openmp_chunks={}
for benchmark in openmp_chunks_benchmarks:  
    d_openmp_chunks_all[benchmark]={}
    d_openmp_chunks[benchmark]={}
    for c in chunk_sizes:
        d_openmp_chunks_all[benchmark][c]={}
        d_openmp_chunks[benchmark][c]={}
        for b in block_sizes:
            d_openmp_chunks_all[benchmark][c][b]={}
            d_openmp_chunks[benchmark][c][b]={}
            for th in thr:
                d_openmp_chunks_all[benchmark][c][b][th]={}
                d_openmp_chunks[benchmark][c][b][th]={}
                for r in repeats:
                    d_openmp_chunks_all[benchmark][c][b][th][r]={}        
                                        
data_files.sort()        
for filename in data_files:    
    if 'openmp' in filename:
        
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        chunk=int(filename.split('/')[-1].split('-')[4]) 
        if option ==1:
            block=int(filename.split('/')[-1].split('-')[5][0:-4])       
        elif option ==2:
            block=filename.split('/')[-1].split('-')[5]+'-'+filename.split('/')[-1].split('-')[6][0:-4]     

        if block in block_sizes:  
            size=[]
            mflops=[]    
            for r in result:        
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
            
            d_openmp_chunks_all[benchmark][chunk][block][th][repeat]['size']=size
            d_openmp_chunks_all[benchmark][chunk][block][th][repeat]['mflops']=mflops
 
    
for benchmark in openmp_chunks_benchmarks:
    for c in chunk_sizes:
        for b in block_sizes:
            for th in thr:
                mflops=[0]*len(size)

                if max(repeats)==1:
                    if 'mflops' in d_openmp_chunks_all[benchmark][c][b][th][1].keys():
                        d_openmp_chunks[benchmark][c][b][th]['size']=size
                        mflops=d_openmp_chunks_all[benchmark][c][b][th][repeats[0]]['mflops']
                        d_openmp_chunks[benchmark][c][b][th]['mflops']=mflops
                else:
                    d_openmp_chunks[benchmark][c][b][th]['size']=size
                    for r in repeats[1:]:
                        mflops=[mflops[i]+d_openmp_chunks_all[benchmark][c][b][th][r]['mflops'][i] for i in range(len(mflops))]                        
                        d_openmp_chunks[benchmark][c][b][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
       
####################################################
#openmp
####################################################        
data_files=glob.glob(openmp_dir+'/*.dat')
d_openmp={}
thr=[]
repeats=[]
openmp_benchmarks=[]
for filename in data_files:
    if 'openmp' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in openmp_benchmarks:
            openmp_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
openmp_benchmarks.sort()      
repeats.sort()      

d_openmp_all={}   

for repeat in repeats:
    d_openmp_all[repeat]={}
    for benchmark in openmp_benchmarks:   
        d_openmp_all[repeat][benchmark]={}
        for th in thr:
            d_openmp_all[repeat][benchmark][th]={}

     

data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    

    if 'openmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
    
            
        for r in result:
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
        
        d_openmp_all[repeat][benchmark][th]['size']=size
        d_openmp_all[repeat][benchmark][th]['mflops']=mflops
 
d_openmp={}
for benchmark in openmp_benchmarks:
    d_openmp[benchmark]={}
    for th in thr:
        d_openmp[benchmark][th]={}
        size_0=d_openmp_all[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        for r in repeats[1:]:
            size=d_openmp_all[r][benchmark][th]['size']
            if size!=size_0:
                print("errorrrrrrrrrrrrrrrrrrrrr")
            mflops=[mflops[i]+d_openmp_all[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]
        d_openmp[benchmark][th]['size']=size_0
        d_openmp[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
        
####################################################
#hpxmp
####################################################        
data_files=glob.glob(hpxmp_dir+'/*.dat')
d_hpxmp={}

hpxmp_benchmarks=[]
for filename in data_files:
    if 'openmp' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpxmp_benchmarks:
            hpxmp_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
hpxmp_benchmarks.sort()      
repeats.sort()      

d_hpxmp_all={}   

for repeat in repeats:
    d_hpxmp_all[repeat]={}
    for benchmark in hpxmp_benchmarks:   
        d_hpxmp_all[repeat][benchmark]={}
        for th in thr:
            d_hpxmp_all[repeat][benchmark][th]={}

     

data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    

    if 'openmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
    
            
        for r in result:
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
        
        d_hpxmp_all[repeat][benchmark][th]['size']=size
        d_hpxmp_all[repeat][benchmark][th]['mflops']=mflops
 
d_hpxmp={}
for benchmark in hpxmp_benchmarks:
    d_hpxmp[benchmark]={}
    for th in thr:
        d_hpxmp[benchmark][th]={}
        size_0=d_hpxmp_all[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        for r in repeats[1:]:
            size=d_hpxmp_all[r][benchmark][th]['size']
            if size!=size_0:
                print(str(th)+"errorrrrrrrrrrrrrrrrrrrrr")
            mflops=[mflops[i]+d_hpxmp_all[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]
        d_hpxmp[benchmark][th]['size']=size_0
        d_hpxmp[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
                
##############################################################################
        #plots
##############################################################################
pp = PdfPages(perf_directory+'/dmatdmatmult_openmp_chunks.pdf')

i=1
benchmark='dmatdmatadd'
for benchmark in openmp_benchmarks:       
    for th in thr:
        plt.figure(i)

        plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label='openmp '+str(th)+' threads')
        plt.plot(d_hpxmp[benchmark][th]['size'], d_hpxmp[benchmark][th]['mflops'],label='hpxmp '+str(th)+' threads')
        plt.plot(d_hpx_before[benchmark][th]['size'], d_hpx_before[benchmark][th]['mflops'],label='hpx '+str(th)+' threads')

        plt.xlabel("# matrix size")           
        plt.ylabel('MFlops')
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(benchmark)
        i=i+1
        plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    plt.show()
    pp.close() 
    i=i+1   
        
for benchmark in hpxmp_benchmarks:       
    plt.figure(i)
    for th in thr:
        plt.plot(d_hpxmp[benchmark][th]['size'], d_hpxmp[benchmark][th]['mflops'],label=str(th)+' threads')
        plt.xlabel("# matrix size")           
        plt.ylabel('MFlops')
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('hpxmp   '+benchmark+'  '+hpxmp_date_str)
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    i=i+1
    pp = PdfPages(perf_directory+'/perf_log_hpxmp_daxpy.pdf')

    th=7
    b='8-512'
    plt.figure(i)
    plt.plot(d_hpx_before[benchmark][th]['size'], d_hpx_before[benchmark][th]['mflops'],label='hpx before changes '+str(th)+' threads', color='red',linestyle='dashed')
    plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label='openmp '+str(th)+' threads',color='black')

    for c in chunk_sizes:   
        if 'mflops' in d_hpx[benchmark][c][b][th].keys():
            plt.plot(d_hpx[benchmark][c][b][th]['size'], d_hpx[benchmark][c][b][th]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')

            plt.xlabel("# matrix size")           
            plt.ylabel('MFlops')
            plt.xscale('log')
            plt.grid(True, 'both')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.title('hpx   '+benchmark+'  '+hpx_date_str)
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    i=i+1
    for th in [14,16]:
        for a in ['4','8','16','32','64','128','256','512','1024']:
            plt.figure(i)
            plt.figure(i)
            plt.plot(d_hpx_before[benchmark][th]['size'], d_hpx_before[benchmark][th]['mflops'],label='hpx before changes '+str(th)+' threads', color='red',linestyle='dashed')
            plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label='openmp '+str(th)+' threads',color='black')

            for b in block_sizes:
                if b.startswith(a+'-'):
                    for c in chunk_sizes:   
                        if 'mflops' in d_hpx[benchmark][c][b][th].keys():
                            plt.plot(d_hpx[benchmark][c][b][th]['size'], d_hpx[benchmark][c][b][th]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
                            plt.xlabel("# matrix size")           
                            plt.ylabel('MFlops')
                            plt.xscale('log')
                            plt.grid(True, 'both')
                            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                            plt.title('hpx   '+benchmark+'  '+hpx_date_str)
            i=i+1    
            plt.savefig(pp, format='pdf',bbox_inches='tight')
            print('')
            
    plt.show()
    pp.close()     
    plt.figure(i)
    for th in [16]:
        plt.plot(d_hpx_before[benchmark][th]['size'], d_hpx_before[benchmark][th]['mflops'],label=str(th)+' threads')
        plt.xlabel("# matrix size")           
        plt.ylabel('MFlops')
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('hpx   before changes '+benchmark+'  '+date_str)
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    i=i+1
        
plt.show()
pp.close() 
       

for th in [16]:
    for a in ['8','16','32']:
        
        for b in block_sizes:
            if b.startswith(a+'-'):
                for c in chunk_sizes:   
                    if 'mflops' in d_openmp_chunks[benchmark][c][b][th].keys() and 'mflops' in d_hpx[benchmark][c][b][th].keys():
                        plt.figure(i)

                        plt.plot(d_hpx[benchmark][c][b][th]['size'], d_hpx[benchmark][c][b][th]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
                        plt.plot(d_openmp_chunks[benchmark][c][b][th]['size'], d_openmp_chunks[benchmark][c][b][th]['mflops'],label='openmp chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads', linestyle='dashed')
                        plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label='openmp '+str(th)+' threads', color='black',linestyle='dashed')
                        plt.plot(d_hpx_before[benchmark][th]['size'], d_hpx_before[benchmark][th]['mflops'],label='hpx '+str(th)+' threads',color='b',linestyle='dashed')

                        plt.xlabel("# matrix size")           
                        plt.ylabel('MFlops')
                        plt.xscale('log')
                        plt.grid(True, 'both')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                        plt.title('hpx   '+benchmark+'  '+hpx_date_str)
                        i=i+1    
                        plt.savefig(pp, format='pdf',bbox_inches='tight')
                        print('')        
plt.show()
pp.close()  
import pickle
#################################
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
##################################       
        
save_obj(d_openmp, '/home/shahrzad/repos/Blazemark/data/matrix/dmatdmatadd/npy/openmp_01-17-2019-1242')

save_obj(d_hpx_before,'/home/shahrzad/repos/Blazemark/data/matrix/dmatdmatadd/npy/hpx_ref_01-06-2019-1059')
a=load_obj('/home/shahrzad/repos/Blazemark/data/matrix/dmatdmatadd/npy/hpx_01-16-2019-0955')

pp = PdfPages(perf_directory+'/perf_log_1-row'+date_str+'.pdf')

c=10
for th in thr:
    for a in [1, 4, 8]:
        for b in d_hpx[benchmark][c].keys():
            if b.startswith(str(a)):
                plt.figure(i)
                plt.plot(d_hpx[benchmark][c][b][th]['size'], d_hpx[benchmark][c][b][th]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
            
                plt.xlabel("# matrix size")           
                plt.ylabel('MFlops')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.title(benchmark)
        i=i+1
        print('')
        plt.savefig(pp, format='pdf',bbox_inches='tight')

        
for th in thr:
    for a in [512, 1024, 2048, 4096]:
        for b in d_hpx[benchmark][c].keys():
            if b.endswith('-'+str(a)):
                plt.figure(i)
                plt.plot(d_hpx[benchmark][c][b][th]['size'], d_hpx[benchmark][c][b][th]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
            
                plt.xlabel("# matrix size")           
                plt.ylabel('MFlops')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.title(benchmark)
        i=i+1        
        print('')
        plt.savefig(pp, format='pdf',bbox_inches='tight')
plt.show()
pp.close()    

    
for b in block_sizes:
    if b =='8-512':
        b='8-512'
        pp = PdfPages(perf_directory+'/perf_log_hpx_aggregated_16'+date_str+'.pdf')

        for c in [4, 10, 15, 20]:   
            plt.figure(i)
            for th in [16]:
#                if 'mflops' in d_hpx_old[benchmark][c][b][th].keys():
#                    plt.plot(d_hpx_old[benchmark][c][b][th]['size'], d_hpx_old[benchmark][c][b][th]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
                plt.plot(d_hpx[benchmark][c][b][th]['size'], d_hpx[benchmark][c][b][th]['mflops'],label='hpx aggregated chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
                plt.plot(a[benchmark][c][b][th]['size'], a[benchmark][c][b][th]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')

                plt.xlabel("# matrix size")           
                plt.ylabel('MFlops')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.title('hpx   '+benchmark+'  '+hpx_date_str)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.savefig(pp, format='pdf',bbox_inches='tight')

            i=i+1  
            print('')
        plt.show()
        pp.close()
c=20
b='8-512'                    
plt.plot(d_hpx[benchmark][c][b][th]['size'], d_hpx[benchmark][c][b][th]['mflops'],label='adaptive chunk_size block_size: '+str(b)+ '  '+str(th)+' threads',color='black')
plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label='openmp '+str(th)+' threads', color='black',linestyle='dashed')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#########################################################
#hpxmp
######################################################
from collections import OrderedDict
linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    
linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('2',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('1',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('0',          (0, (3, 5, 1, 5))),
     ('3',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
i=1    
benchmark='dmatdmatadd'
benchmark='dmatdmatmult'

for benchmark in hpxmp_benchmarks:   
    for th in [4,8,16]:
        plt.figure(i)    
        pp = PdfPages(perf_directory+'/hpxmp/figures/scale/'+benchmark+'_'+str(th)+'.pdf')
        print(perf_directory+'/hpxmp/figures/scale/'+benchmark+'_'+str(th)+'.pdf')
        plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label='openmp '+str(th)+' threads',color='black')
        plt.plot(d_hpxmp[benchmark][th]['size'], d_hpxmp[benchmark][th]['mflops'],label='hpxmp '+str(th)+' threads',color='black',linestyle='dashed')
        plt.xlabel("size $n$")           
        plt.ylabel('MFlops')
        plt.xscale('log')
        plt.grid(True, 'both')
        if benchmark=='daxpy' or benchmark=='dvecdvecadd':
            plt.legend(loc=1)
        else:
            plt.legend(loc=2)
#        plt.title(benchmark)
        plt.savefig(pp, format='pdf',bbox_inches='tight')
#        plt.savefig(perf_directory+'/'+benchmark+'_'+str(th)+'.png')
        i=i+1
        plt.show()
        pp.close() 

 
from pylab import *
set_cmap('Blues')
import cv2

for benchmark in hpxmp_benchmarks:
    if benchmark=='daxpy' or benchmark=='dvecdvecadd':
        sizes=[43794, 77580, 103258, 431318, 764102, 1017019, 2180065, 4248326, 7526167, 10000000]
    elif benchmark=='dmatdvecmult':
        sizes=[414, 605, 731, 972, 1175, 2079, 3041, 7165, 10000 ]
    else:
        sizes=[230, 300, 455, 523, 600, 793, 884, 1048, 2100, 3193, 7000]
    pp = PdfPages(perf_directory+'/hpxmp/figures/heatmap/'+benchmark+'_heatmap.pdf')

    fig, ax = plt.subplots()
    thr=np.arange(1,17).tolist()
    all_sizes=d_openmp[benchmark][th]['size']
    hpxmp=np.zeros((len(sizes),len(thr)))
    
    for i in range(len(sizes)):
        for j in range(len(thr)):
            hpxmp[i,j]=round(d_hpxmp[benchmark][thr[j]]['mflops'][all_sizes.index(sizes[i])]/d_openmp[benchmark][thr[j]]['mflops'][all_sizes.index(sizes[i])],1)
    #hpxmp=hpxmp[s,:]
    im = ax.imshow(hpxmp)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("ratio $r$", rotation=-90, va="bottom")
    ax.set_xlabel('number of threads')
    ax.set_ylabel('size $n$')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(thr)))
    ax.set_yticks(np.arange(len(sizes)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(thr)
    ax.set_yticklabels(sizes)
    #ax.set_yticklabels(list(sizes[p] for p in s))
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(sizes)):
        for j in range(len(thr)):
            text = ax.text(j, i, hpxmp[i,j], ha="center", va="center", color="black")
    
#    ax.set_title("hpxmp speedup")
    fig.tight_layout()
    
    fname=perf_directory+'/hpxmp/figures/heatmap/'+benchmark+'_heatmap.png'
#    plt.savefig(fname,bbox_inches='tight')
#    img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
#    cv2.imwrite(fname,img)
    plt.savefig(pp, format='pdf',bbox_inches='tight')

    plt.show()
    pp.close() 