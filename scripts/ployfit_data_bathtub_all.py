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

filename_ws='/home/shahrzad/repos/Blazemark/data/data_perf_all_ws.csv'
filename='/home/shahrzad/repos/Blazemark/data/data_perf_all.csv'

perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/bathtub3d'
benchmarks=['dmatdmatadd']
plot=True
error=False
save=False
node='marvin'
benchmark='dmatdmatadd'
collect_3d_data=True
build_model=False
plot_type='params_th'
runtime='hpx'
#runtime='openmp'

def remove_duplicates(array):
    g=array[:,0]
    p=array[:,-1]
    g_dict={}
    if np.shape(array)[1]==2:
        for i in range(len(g)):
            if g[i] not in g_dict.keys():
                g_dict[g[i]]=p[i]
            else:
                g_dict[g[i]]+=p[i]
        p=np.asarray([g_dict[gd]/Counter(g)[gd] for gd in g_dict.keys()])
        g=np.asarray([gd for gd in g_dict.keys()])
        array=np.zeros((np.shape(p)[0],2))
        array[:,0]=g
        array[:,1]=p
    else:
        g=array[:,-2]
        p=array[:,-1]
        t=array[:,1]
        nt=array[:,0]
        
        count={}
        for i in range(len(g)):
            if (g[i],t[i],nt[i]) not in g_dict.keys():
                g_dict[(g[i],t[i],nt[i])]=p[i]
                count[(g[i],t[i],nt[i])]=1
            else:
                g_dict[(g[i],t[i],nt[i])]+=p[i]                
                count[(g[i],t[i],nt[i])]+=1
        p=np.asarray([g_dict[gd]/count[gd] for gd in g_dict.keys()])
        g=np.asarray([gd[0] for gd in g_dict.keys()])
        t=np.asarray([gd[1] for gd in g_dict.keys()])
        nt=np.asarray([gd[2] for gd in g_dict.keys()])

        array=np.zeros((np.shape(p)[0],3))
        array[:,0]=nt
        array[:,1]=t
        array[:,2]=g
        array[:,3]=p
    return array

def grain_dict(array,avg=False):
    g_dict={}
    
    g=array[:,-2]
    p=array[:,-1]
    t=array[:,1]
    nt=array[:,0]
    
    for i in range(len(g)):
        if g[i] not in g_dict.keys():
            g_dict[g[i]]={}
        if t[i] not in g_dict[g[i]].keys():
            g_dict[g[i]][t[i]]=[[],[]]
        g_dict[g[i]][t[i]][0].append(p[i])
        g_dict[g[i]][t[i]][1].append(nt[i])

    if avg:
        for gd in g_dict.keys():
            for td in g_dict[gd].keys():
                g_dict[gd][td][0]=sum(g_dict[gd][td][0])/len(g_dict[gd][td][0])
                g_dict[gd][td][1]=np.ceil(sum(g_dict[gd][td][1])/len(g_dict[gd][td][1]))
    return g_dict

def find_max_range(filename,benchmarks=None,plot=True,error=False,save=False,perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/polynomial'
,plot_type='perf_curves',collect_3d_data=False,build_model=False):
    titles=['runtime','node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','work_per_core','w1','w2','w3','w4','w5','w6','w7','w8','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops','include']

    ranges={}
    deg=2
    dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
    for col in titles[3:]:
        dataframe[col] = dataframe[col].astype(float)
    i=1  
    h=1
    k=1
    nodes=dataframe['node'].drop_duplicates().values
    nodes.sort()
    original_benchmarks=benchmarks
    modeled_params={}
    g_params={}
    threads={}
    included=dataframe['include']==1

    for node in nodes:
        node_selected=dataframe['node']==node
        df_n_selected=dataframe[node_selected & included]

        g_params[node]={}
        if original_benchmarks is None:
            benchmarks=df_n_selected['benchmark'].drop_duplicates().values
        benchmarks.sort()
        threads[node]={}
        for benchmark in benchmarks:             
            g_params[node][benchmark]={}
            benchmark_selected=dataframe['benchmark']==benchmark
            rt_selected=dataframe['runtime']==runtime
            num_threads_selected=dataframe['num_threads']<=8
            df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected & rt_selected]         
            matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
            matrix_sizes.sort()
            thr=df_nb_selected['num_threads'].drop_duplicates().values
            thr.sort()
            threads[node][benchmark]=thr
            thr=thr[1:]
            matrix_sizes=matrix_sizes          

            i=1
            for m in matrix_sizes:
                simdsize=4.
                if node=='medusa':
                    simdsize=8.

                aligned_m=m
                if m%simdsize!=0:
                    aligned_m=m+simdsize-m%simdsize
                if benchmark=='dmatdmatadd':                            
                    mflop=(aligned_m)*m                           
                elif benchmark=='dmatdmatdmatadd':
                    mflop=2*(aligned_m)*m
                else:
                    mflop=2*(aligned_m)**3        
                
                m_selected=df_nb_selected['matrix_size']==m
#                block_selected_r=df_nb_selected['block_size_row']==4
#                block_selected_c=df_nb_selected['block_size_col']!=64
#                df_nb_selected=df_nb_selected[ block_selected_r]# | block_selected_c]
                features=['num_tasks','num_threads','work_per_core','chunk_size','num_blocks','w1','w2','w3','w4','w5','w6','w7','w8','grain_size','execution_time']
#                features=['num_tasks','num_threads','execution_time']


                df_selected=df_nb_selected[m_selected][features]

#                    df_selected=df_nb_selected[m_selected & th_selected & block_selected_r & block_selected_c][features]

                array=df_selected.values
                array=array.astype(float)
  
                a_s=np.argsort(array[:,0])
                
                array=array[a_s]
                g_params[node][benchmark]=grain_dict(array,1)
#                array=remove_duplicates(array)
                    
                data_size=np.shape(array)[0]
                per=[i for i in range(data_size) if i%3!=2]

                train_size=len(per)

#                        train_size=int(np.ceil(0.6*data_size))
                test_size=data_size-train_size
#                        per = np.random.permutation(data_size)
#                        train_set=array[per[0:train_size],:-1] 
#                        train_labels=array[per[0:train_size],-1]  
#                        test_set=array[per[train_size:],:-1]  
#                        test_labels=array[per[train_size:],-1]  
                train_set=array[per,:-1]  
                train_labels=array[per,-1]  

                test_items=[item for item in np.arange((data_size)) if item not in per]
                test_set=array[test_items,:-1]  
                test_labels=array[test_items,-1] 

                def my_func_3d_3_no_ws(data,alpha,gamma,q): 
                    ts=g_params[node][benchmark][mflop][1][0]
                    kappa=0.
                    N=data[:,1]
                    n_t=data[:,0]
                    n_b=data[:,4]
                    w_c=data[:,2]
                    c=data[:,3]
                    g=data[:,-1]
                    M=np.minimum(n_t,N) 
                    L=np.ceil(n_t/(N))
                    w_all=data[:,5:13].copy()
                    for j in range(np.shape(data)[0]):
                        for i in range(int(N[j])):
                            if n_t[j]<=N[j]:
                                w_all[j,i]=abs(g[j]-w_all[j,i])
                            else:
                                w_all[j,i]=0
                    ws=np.sum(w_all[:,],axis=1)/g
#                    ts=476.
#                    M=np.ceil(N-np.log(1+(np.exp(N)-1)*np.exp(-n_t)))
#                    return alpha*d/M+ts*(1-q)+ts*q/M
#                    return alpha*n_t/M+ts/M+ts*q*(M-1)/M+ts*gamma*(M-1)+w*(c/(n_b%c+0.01))*g
#                    return alpha*L+(ts+ts*gamma*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop+q+(d*(M))*np.heaviside(N-n_t,1)+h*n_t*np.heaviside(n_t-N,1)
#                    return alpha*L+(ts+ts*gamma*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop+q+h*n_t*np.heaviside(n_t-N,1)+(d/N)*((n_t-1)%N)*np.heaviside(N-n_t,1)+(d1/N)*((n_t-1)%N)*np.heaviside(n_t-N-1,1)
                    return q+alpha*L+(ts+ts*gamma*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop#+h*n_t*np.heaviside(n_t-N,1)+p*np.heaviside(n_t-N,0)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)
                    

                def my_func_3d_3(data,alpha,gamma,d,h,q): 
#                    ts=g_params[node][benchmark][mflop][1][0]
                    kappa=0.
                    N=data[:,1]
                    n_t=data[:,0]
                    n_b=data[:,4]
                    w_c=data[:,2]
                    c=data[:,3]
                    g=data[:,-1]
                    M=np.minimum(n_t,N) 
                    L=np.ceil(n_t/(N))
                    w_all=data[:,5:13].copy()
                    for j in range(np.shape(data)[0]):
                        for i in range(int(N[j])):
                            if n_t[j]<=N[j]:
                                w_all[j,i]=abs(g[j]-w_all[j,i])
                            else:
                                w_all[j,i]=0
                    ws=np.sum(w_all[:,],axis=1)/g
#                    ts=476.
#                    M=np.ceil(N-np.log(1+(np.exp(N)-1)*np.exp(-n_t)))
#                    return alpha*d/M+ts*(1-q)+ts*q/M
#                    return alpha*n_t/M+ts/M+ts*q*(M-1)/M+ts*gamma*(M-1)+w*(c/(n_b%c+0.01))*g
#                    return alpha*L+(ts+ts*gamma*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop+q+(d*(M))*np.heaviside(N-n_t,1)+h*n_t*np.heaviside(n_t-N,1)
#                    return alpha*L+(ts+ts*gamma*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop+q+h*n_t*np.heaviside(n_t-N,1)+(d/N)*((n_t-1)%N)*np.heaviside(N-n_t,1)+(d1/N)*((n_t-1)%N)*np.heaviside(n_t-N-1,1)
#                    return q*N+alpha*L+(ts+ts*(gamma)*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop+h*n_t*np.heaviside(n_t-N,1)+(d/N)*((n_t-1)%N)*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)
                    ts=g_params[node][benchmark][mflop][1][0]
                    ps=ts
                    return q*(N-1)*(N-2)/(mflop)+alpha*L+(ts+ts*gamma*(M-1))*(w_c)/mflop+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/(mflop*N))*((n_t-1))*np.heaviside(N-n_t,1)#+h*n_t*np.heaviside(n_t-N,1)+p*np.heaviside(n_t-N,0)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)
                    

                def my_func_3d_4(data,alpha,gamma,d,h,q): 
#                    ts=g_params[node][benchmark][mflop][1][0]
                    kappa=0.
                    N=data[:,1]
                    n_t=data[:,0]
                    n_b=data[:,4]
                    w_c=data[:,2]
                    c=data[:,3]
                    g=data[:,-1]
                    M=np.minimum(n_t,N) 
                    L=np.ceil(n_t/(M))
                    w_all=data[:,5:13].copy()
                    for j in range(np.shape(data)[0]):
                        for i in range(int(N[j])):
                            if n_t[j]<=N[j]:
                                w_all[j,i]=abs(g[j]-w_all[j,i])
                            else:
                                w_all[j,i]=0
                    ws=np.sum(w_all[:,],axis=1)/g
#                    ts=476.
#                    M=np.ceil(N-np.log(1+(np.exp(N)-1)*np.exp(-n_t)))
#                    return alpha*d/M+ts*(1-q)+ts*q/M
#                    return alpha*n_t/M+ts/M+ts*q*(M-1)/M+ts*gamma*(M-1)+w*(c/(n_b%c+0.01))*g
#                    return alpha*L+(ts+ts*gamma*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop+q+(d*(M))*np.heaviside(N-n_t,1)+h*n_t*np.heaviside(n_t-N,1)
#                    return alpha*L+(ts+ts*gamma*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop+q+h*n_t*np.heaviside(n_t-N,1)+(d/N)*((n_t-1)%N)*np.heaviside(N-n_t,1)+(d1/N)*((n_t-1)%N)*np.heaviside(n_t-N-1,1)
#                    return q*N+alpha*L+(ts+ts*(gamma)*(M-1)+ts*kappa*M*(M-1))*(w_c)/mflop+h*n_t*np.heaviside(n_t-N,1)+(d/N)*((n_t-1)%N)*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)
                    ts=g_params[node][benchmark][mflop][1][0]
                    return q*(N-1)*(N-2)/(mflop)+alpha*L+(ts+ts*gamma*(M-1))*(w_c)/mflop+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/(mflop*N))*((n_t-1))*np.heaviside(N-n_t,1)#+h*n_t*np.heaviside(n_t-N,1)+p*np.heaviside(n_t-N,0)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)
                    
    
                
 
                param_bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,np.inf])
                popt3, pcov=curve_fit(my_func_3d_3,train_set,train_labels,method='trf',bounds=param_bounds)
                popt4, pcov=curve_fit(my_func_3d_4,train_set,train_labels,method='trf',bounds=param_bounds)
#
#                popts.append(popt3)
#
#                ts=g_params[node][benchmark][mflop][1][0]
#
#
#                version=5
#                pp = PdfPages(perf_directory+'/'+str(version)+'/'+runtime+'_'+node+'_'+benchmark+'_'+str(int(m))+'_bathtub3d_mflops.pdf')

                for th in range(1,9):          
                    new_array=test_set[test_set[:,1]==th]
#                    z1=mflop/my_func_3d_1(new_array,*popt1)
#                    z2=mflop/my_func_3d_2(new_array,*popt2)
                    z3=my_func_3d_3(new_array,*popt3)
                    [alpha,gamma,d,h,q]=popt3
                    popt4=[alpha,0,0,0,0]
#                    z4=my_func_3d_4(new_array,*popt4)

#                    z6=mflop/my_func_3d_3(new_array,*popt_i)

#                    z6=mflop/my_func_3d_6(new_array,*popt6)

##                    z5=mflop/my_func_3d_3(new_array,*[2*ts,alpha,gamma,q,d,h,d1])
#                    z4=mflop/my_func_3d_4(new_array,*popt4)
#                    z5=mflop/(ts*my_func_3d_5(new_array,*popt5))
                    ts=g_params[node][benchmark][mflop][1][0]
                    plt.figure(i)
                    plt.axes([0, 0, 2, 1])
                    plt.scatter(new_array[:,-1],test_labels[test_set[:,1]==th],color='blue',label='true',marker='.')
                    
    #                    plt.scatter(new_array[:,-1],z1,label='pred1',marker='.')
    #                    plt.scatter(new_array[:,-1],z2,label='pred2',marker='.')
                    plt.scatter(new_array[:,-1],z3,label='pred3',marker='.',color='red')
#                    plt.scatter(new_array[:,-1],z6,label='pred6',marker='.',color='gray')
#                    plt.scatter(mflop/new_array[:,-1],z4/ts,label='pred4',marker='.',color='green')


#                    plt.scatter(new_array[:,-1],z5,label='pred5',marker='.',color='red')
#                    plt.scatter(new_array[:,-1],z3,label='pred from dmatdmatadd',marker='.',color='green')
#                    plt.axvline(mflop/th,color='gray',linestyle='dotted')    
#                    plt.axvline((mflop/(th*(th+1)))+(0.01/(th+1)),color='green')
#                    plt.axvline(np.sqrt(popt_3[0]*mflop/(0.1*th)),color='purple')
    #                plt.scatter(new_array[:,-1],z4,label='pred4',marker='.',color='orange')
                    plt.grid(True,'both')
                    plt.xscale('log')
                    plt.xlabel('problem_size/grain_size')
                    plt.ylabel('1/speedup')
                    plt.title('test set  matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
            
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    plt.savefig(perf_dir+node+'/'+node+'_'+benchmark+'_'+str(int(m))+'_'+str(int(th))+'.png',bbox_inches='tight')

                    i=i+1
#                    plt.savefig(pp,format='pdf',bbox_inches='tight')
                plt.show()
                pp.close()

#            np.save(perf_directory+'/'+str(version)+'/'+runtime+'_'+node+'_'+benchmark+'_popts_'+str(version)+'.npy',popts)

perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hxp_for_loop/1/blaze/'

