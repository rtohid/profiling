#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:46:28 2020

@author: shahrzad
"""
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
runtime='hpx'
node='marvin'

benchmark='dmatdmatadd'
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/bathtub3d'

simdsize=4
m=4222
aligned_m=m
if m%simdsize!=0:
    aligned_m=m+simdsize-m%simdsize
#                    benchmark='dmatdmatadd'
if benchmark=='dmatdmatadd':                            
    mflop=(aligned_m)*m                           
elif benchmark=='dmatdmatdmatadd':
    mflop=2*(aligned_m)*m
else:
    mflop=2*(aligned_m)**3   


np.save('/home/shahrzad/repos/Blazemark/data/array_all_'+str(int(m))+'.npy',array)
np.save('/home/shahrzad/repos/Blazemark/data/array_no_ws_'+str(int(m))+'.npy',array)

array=np.load('/home/shahrzad/repos/Blazemark/data/array_all_'+str(int(m))+'.npy')
array_off=np.load('/home/shahrzad/repos/Blazemark/data/array_no_ws_'+str(int(m))+'.npy')


#Mflops based on grain size
pp = PdfPages(perf_directory+'/'+runtime+'_'+node+'_'+benchmark+'_'+str(int(m))+'_bathtub3d_mflops_work_stealing.pdf')
i=1
for th in range(1,9):
    plt.figure(i)
    plt.axes([0, 0, 1.8, 1])

    plt.scatter(array[array[:,1]==th][:,-2],mflop/array[array[:,1]==th][:,-1],label='work stealing on',marker='.')
    plt.scatter(array_off[array_off[:,1]==th][:,-2],mflop/array_off[array_off[:,1]==th][:,-1],label='work stealing off',marker='.')
    plt.axvline(mflop/th,color='gray',linestyle='dotted')    
    plt.xlabel('Grain size')
    plt.ylabel('Mflops')
    plt.title('matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1
#    plt.grid(True,'both')

    plt.savefig(pp,format='pdf',bbox_inches='tight')    
plt.show()
pp.close()


#Mflops based on num tasks   
i=1
for th in range(1,9):
    plt.figure(i)
    plt.axes([0, 0, 1.8, 1])

    plt.scatter(array[array[:,1]==th][:,0],mflop/array[array[:,1]==th][:,-1],label='work stealing on',marker='.')
    plt.scatter(array_off[array_off[:,1]==th][:,0],mflop/array_off[array_off[:,1]==th][:,-1],label='work stealing off',marker='.')
    plt.xlabel('num_tasks')
    plt.ylabel('Mflops')
    plt.title('matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
    plt.xscale('log')
    plt.axvline(th,color='gray',linestyle='dotted')    

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1
#    plt.grid(True,'both')


#Execution time based on grain size
i=1
for th in range(1,9):
    plt.figure(i)
    plt.axes([0, 0, 1.8, 1])

    plt.scatter(array[array[:,1]==th][:,-2],array[array[:,1]==th][:,-1],label='work stealing on',marker='.')
    plt.scatter(array_off[array_off[:,1]==th][:,-2],array_off[array_off[:,1]==th][:,-1],label='work stealing off',marker='.')
    plt.axvline(mflop/th,color='gray',linestyle='dotted')    
    plt.xlabel('Grain size')
    plt.ylabel('Execution time')
    plt.title('matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1


#Execution time based on num tasks
i=1
for th in range(1,9):
    plt.figure(i)
    plt.axes([0, 0, 1.8, 1])

    plt.scatter(array[array[:,1]==th][:,0],array[array[:,1]==th][:,-1],label='work stealing on',marker='.')
    plt.scatter(array_off[array_off[:,1]==th][:,0],array_off[array_off[:,1]==th][:,-1],label='work stealing off',marker='.')
    plt.axvline(th,color='gray',linestyle='dotted')    
    plt.xlabel('num_tasks')
    plt.ylabel('Execution time')
    plt.title('matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1

  