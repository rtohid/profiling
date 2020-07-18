#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:24:26 2020

@author: shahrzad
"""
import glob

directory='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/idle_cores/split_info'
thr=[]
data_files=[]

[data_files.append(i) for i in glob.glob(directory+'/*.dat')]

chunk_sizes=[]
num_iterations=[]
iter_lengths=[]
nodes=[]
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
    all_task_sizes={}
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
                all_task_sizes[num_iteration]={th:[]}
                task_size=[]
                for r in [r for r in result if r.startswith(' task index')]:  
                    start_ind=r.split('from:')[1].split('to:')[0].strip()
                    end_ind=r.split('to:')[1].split('\n')[0].strip()

                    task_size.append([int(end_ind)-int(start_ind)])
                all_task_sizes[num_iteration][th]=task_size