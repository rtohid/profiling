#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:47:00 2019

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

class my_prediction:
    def create_dict_refernce(self,directory):
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
    
    def polyfit3d(self,x, y, z, d, x_p, y_p, z_p):
        #m**3*g**2
        ncols = (x_p + 1)*(y_p + 1)*(z_p + 1)
        Q = np.zeros((x.size, ncols))
        ijl = itertools.product(range(x_p + 1), range(y_p + 1), range(z_p + 1))
        for k, (i,j,l) in enumerate(ijl):
            Q[:,k] = x**i * y**j * z**l
        m, _, _, _ = np.linalg.lstsq(Q, d)
        return m
    
    def polyval3d(self,x, y, z, x_p, y_p, z_p, m):
        ijl = itertools.product(range(x_p + 1), range(y_p + 1), range(z_p + 1))
        w = np.zeros_like(x)
        for a, (i,j,l) in zip(m, ijl):
            w += a * x**i * y**j * z**l
        return w
    
    def polyval3d_given_yz(self,y, z, x_p, y_p, z_p, m):
        ijl = itertools.product(range(x_p + 1), range(y_p + 1), range(z_p + 1))
        coef=[0.]*(x_p+1)
        for a, (i,j,l) in zip(m, ijl):
            coef[x_p-i] += a * y**j * z**l
        return coef
    
    
    def find_max_range(self,filename,benchmarks=None,plot=True,error=False,save=False,perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/polynomial'
    ,plot_type='perf_curves',collect_3d_data=False,build_model=False):
        titles=['node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','num_elements_uncomplete','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','cache_block','datatype','cost','mflops']
    
        ranges={}
        deg=2
        dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
        for col in titles[2:]:
            dataframe[col] = dataframe[col].astype(float)
        i=1  
        j=1
        k=1
        nodes=dataframe['node'].drop_duplicates().values
        nodes.sort()
        if benchmarks is None:
            benchmarks=dataframe['benchmark'].drop_duplicates().values
        benchmarks.sort()
        modeled_params={}
        m_data={}
        all_data={}
        for node in nodes:
            ranges[node]={}
            m_data[node]={}
            all_data[node]={}
            modeled_params[node]={}
            for benchmark in benchmarks: 
                modeled_params[node][benchmark]={}
                all_data[node][benchmark]={}
                node_selected=dataframe['node']==node
                benchmark_selected=dataframe['benchmark']==benchmark
                num_threads_selected=dataframe['num_threads']<=8
                df_nb_selected=dataframe[node_selected & benchmark_selected & num_threads_selected] 
                matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
                matrix_sizes.sort()
                thr=df_nb_selected['num_threads'].drop_duplicates().values
                thr.sort()
                ranges[node][benchmark]={}
                m_data[node][benchmark]={'matrix_sizes':[],'threads':[]}
                m_data[node][benchmark]['matrix_sizes']=matrix_sizes
                m_data[node][benchmark]['threads']=thr
                m_data[node][benchmark]['params']={}
                if save:
                    perf_filename=perf_directory+'/'+node+'_'+benchmark+'_'+plot_type+'.pdf'
                else:
                    perf_filename=perf_directory+'/'+node+'_'+benchmark+'.pdf'
                pp=''
                if plot and save:
                    pp = PdfPages(perf_filename)
                for th in thr: 
                    if build_model or collect_3d_data:
                        all_data[node][benchmark][th]={}
                        P_train=[]
                        G_train=[]
                        M_train=[]
                        T_train=[]
                        P_test=[]
                        G_test=[]
                        M_test=[]
                        T_test=[]
                    
                    p_range=[0.,np.inf]
                    m_data[node][benchmark]['params'][th]={}
                    
                    for m in matrix_sizes:
                        m_selected=df_nb_selected['matrix_size']==m
                        th_selected=df_nb_selected['num_threads']==th
            
                        features=['grain_size','mflops']
                        df_selected=df_nb_selected[m_selected & th_selected][features]
                    
                        array=df_selected.values
                        array=array.astype(float)
                        
                        array[:,:-1]=np.log10(array[:,:-1])
                        data_size=np.shape(array)[0]
                        m_data[node][benchmark]['params'][th][m]=[0.0]*(deg+1)
                        if data_size>=8:
    
                            per = np.random.permutation(data_size)
                            train_size=int(np.ceil(0.6*data_size))
                            train_set=array[per[0:train_size],:-1]  
                            train_labels=array[per[0:train_size],-1]  
                            test_set=array[per[train_size:],:-1]  
                            test_labels=array[per[train_size:],-1]  
                            test_size=data_size-train_size
                            z = np.polyfit(train_set[:,0], train_labels, deg)
    #                        print(th,m,z)
                            p = np.poly1d(z)
                            m_data[node][benchmark]['params'][th][m]=z
    
                            if error:
                                s=0
                                for i in range(test_size):
                                    s=s+(test_labels[i]-p(test_set[i]))**2
                                s=np.sqrt(s/test_size)
                                print(test_size,'estimated standard error:'+str(s))
    #                        A=np.argsort(train_set[:,0])
    #                        g=np.asarray([train_set[a,0] for a in A])
    #                        mf=np.asarray([train_labels[a] for a in A])
    #                        
    #                        plt.plot(g, mf, label='training data')
                            if build_model or collect_3d_data:
                                for j in range(train_size):
                                    G_train.append(train_set[j,0])
                                    P_train.append(train_labels[j])
                                    M_train.append(float(m))
                                    T_train.append(float(th))
                                for j in range(data_size-train_size):
                                    G_test.append(test_set[j,0])
                                    P_test.append(test_labels[j])
                                    M_test.append(float(m))
                                    T_test.append(float(th))
                            max_perf=np.asarray(-z[1]/(2*z[0]))    
                            y0=p(max_perf)
                            new_eq=[z[0],z[1],z[2]-0.9*y0] 
                            x0=(-new_eq[1]+np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
                            x1=(-new_eq[1]-np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])   
                            if min(x0,x1)>p_range[0]:
                                p_range[0]=min(x0,x1)
                            if max(x0,x1)<p_range[1]:
                                p_range[1]=max(x0,x1)  
                            if plot:
                                if plot_type=='params_th' or plot_type=='all':
                                    plt.figure(i)
                                    plt.scatter(m,z[0])
                                    plt.xlabel('matrix size')
                                    plt.ylabel('z[0]')   
                                    plt.grid(True, 'both')
                                    plt.title(benchmark+'  '+str(int(th))+' threads')
                                    plt.figure(i+1)
                                    plt.scatter(m,z[1])
                                    plt.xlabel('matrix size')
                                    plt.ylabel('z[1]')  
                                    plt.grid(True, 'both')
                                    plt.title(benchmark+'  '+str(int(th))+' threads')
                                    plt.figure(i+2)
                                    plt.scatter(m,z[2])
                                    plt.xlabel('matrix size')
                                    plt.ylabel('z[2]')  
                                    plt.grid(True, 'both')
                                    plt.title(benchmark+'  '+str(int(th))+' threads')
                                elif plot_type=='perf_curves' or plot_type=='all':    
                                    plt.figure(j)
                                    plt.plot([x0,x1],[p(x0),p(x1)],marker='+',label='interval of max performance  matrix size:'+str(m))
                        #            plt.scatter(max_perf,p(max_perf),marker='o',color='r')
                                
                                    A=np.argsort(test_set[:,0])
                                    g=np.asarray([test_set[a,0] for a in A])
                                    mf=np.asarray([test_labels[a] for a in A])
                                    plt.plot(g, p(g), label='test set fitted with degree '+str(deg)+'  matrix size:'+str(m))
                        #            plt.plot(g,mf,label='test set real')
                                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                    plt.title(node+'  '+benchmark+'  '+str(th)+' threads')
                                    plt.grid(True, 'both')
                                    plt.xlabel('grain size')
                                    plt.ylabel('MFlops')   
                    if plot and save:
                        plt.savefig(pp,format='pdf',bbox_inches='tight')
    
                    ranges[node][benchmark][th]=[10**p_range[0], 10**p_range[1]]
                    z0_t=[m_data[node][benchmark]['params'][th][m][0] for m in matrix_sizes]
                    n=np.polyfit(matrix_sizes,z0_t,3)
                    p1 = np.poly1d(n)
    #                modeled_params[node][benchmark][th]['z0']=n
                    if plot and (plot_type=='params_th' or plot_type=='all'):
                        plt.figure(i)
                        plt.plot(matrix_sizes,p1(matrix_sizes))
                        plt.xlabel('matrix size')
                        plt.ylabel('z[0]')   
                        plt.grid(True, 'both')
                        plt.title(benchmark+'  '+str(int(th))+' threads')
                    z1_t=[m_data[node][benchmark]['params'][th][m][1] for m in matrix_sizes]
                    n=np.polyfit(matrix_sizes,z1_t,3)
                    p2 = np.poly1d(n)
    #                modeled_params[node][benchmark][th]['z1']=n
                    if plot and (plot_type=='params_th' or plot_type=='all'):
                        plt.figure(i+1)
                        plt.plot(matrix_sizes,p2(matrix_sizes))
                        plt.xlabel('matrix size')
                        plt.ylabel('z[1]')   
                        plt.grid(True, 'both')
                        plt.title(benchmark+'  '+str(int(th))+' threads')
                    z2_t=[m_data[node][benchmark]['params'][th][m][2] for m in matrix_sizes]
                    n=np.polyfit(matrix_sizes,z2_t,3)
                    p3 = np.poly1d(n)
    #                modeled_params[node][benchmark][th]['z2']=n
                    if plot and (plot_type=='params_th' or plot_type=='all'):
                        plt.figure(i+2)
                        plt.plot(matrix_sizes,p3(matrix_sizes))
                        plt.xlabel('matrix size')
                        plt.ylabel('z[2]')   
                        plt.grid(True, 'both')
                        plt.title(benchmark+'  '+str(int(th))+' threads')
                    i=i+3
                    j=j+1
                    if collect_3d_data:
                        all_data[node][benchmark][th]['train']=[G_train,M_train,T_train,P_train]
                        all_data[node][benchmark][th]['test']=[G_test,M_test,T_test,P_test]
    
                if plot and (plot_type=='params_m' or plot_type=='all'):
                    for m in matrix_sizes:     
                        
                        n=np.polyfit(thr,[m_data[node][benchmark]['params'][t][m][0] for t in thr],1)
                        p1 = np.poly1d(n)
    #                    print(m,'z0',n)
                        n=np.polyfit(thr,[m_data[node][benchmark]['params'][t][m][1] for t in thr],1)
                        p2 = np.poly1d(n)
    #                    print(m,'z1',n)
                        n=np.polyfit(thr,[m_data[node][benchmark]['params'][t][m][2] for t in thr],1)
                        p3 = np.poly1d(n)
                        
    #                    print(m,'z2',n)
                        for th in thr: 
                            plt.figure(k)
                            plt.scatter(th,m_data[node][benchmark]['params'][th][m][0])
                            plt.plot(thr,p1(thr))
                            plt.xlabel('threads')
                            plt.ylabel('z[0]')  
                            plt.grid(True, 'both')
                            plt.title(benchmark+'  matrix size:'+str(int(m)))
                            plt.figure(k+1)
                            plt.scatter(th,m_data[node][benchmark]['params'][th][m][1])
                            plt.plot(thr,p2(thr))
    
                            plt.xlabel('threads')
                            plt.ylabel('z[1]')  
                            plt.grid(True, 'both')
                            plt.title(benchmark+'  matrix size:'+str(int(m)))
                            plt.figure(k+2)
                            plt.scatter(th,m_data[node][benchmark]['params'][th][m][2])
                            plt.plot(thr,p3(thr))
    
                            plt.xlabel('threads')
                            plt.ylabel('z[2]')  
                            plt.grid(True, 'both')
                            plt.title(benchmark+'  matrix size:'+str(int(m)))
                        k=k+3
                if plot and save:
                    plt.show()
                    pp.close()
                if build_model:
                    x=[]
                    y=[]
                    z=[]
                    d=[]
                    for th in thr:
                        [x.append(i) for i in all_data[node][benchmark][th]['train'][0]]
                        [y.append(i) for i in all_data[node][benchmark][th]['train'][1]]
                        [z.append(i) for i in all_data[node][benchmark][th]['train'][2]]
                        [d.append(i) for i in all_data[node][benchmark][th]['train'][3]]
                    x=np.asarray(x)
                    y=np.asarray(y)
                    z=np.asarray(z)
                    d=np.asarray(d)
                    x_p=2
                    y_p=3
                    z_p=3
                    modeled_params[node][benchmark]=polyfit3d(x, y, z, d, x_p, y_p, z_p)
        if collect_3d_data or build_model:        
            return ranges,m_data,thr,modeled_params,all_data
        else:
            return ranges,m_data,thr,modeled_params
        
        
    def select_chunk_block(self,data,th,m,node,benchmark,block_size_row,block_size_col=None,plot=False,from_formula=False):
        if block_size_col is None:
            block_size_col=float(m)
        if not from_formula: 
            ranges=data[0]
            return [int(np.ceil(ranges[node][benchmark][th][0]/(block_size_row*block_size_col))),int(np.floor(ranges[node][benchmark][th][1]/(block_size_row*block_size_col)))]
        else:
            g_p=2
            m_p=3
            t_p=3
            z=polyval3d_given_yz(m, th, g_p, m_p, t_p, data[3][node][benchmark])
    #        modeled_params=data[-1]
    #        z_m_0=modeled_params[node][benchmark][th]['z0']
    #        z_m_1=modeled_params[node][benchmark][th]['z1']
    #        z_m_2=modeled_params[node][benchmark][th]['z2']
            #z=[np.poly1d(z_m_0)(m),np.poly1d(z_m_1)(m),np.poly1d(z_m_2)(m)]
            p=np.poly1d(z)
            max_perf=np.asarray(-z[1]/(2*z[0]))    
            y0=p(max_perf)
            new_eq=[z[0],z[1],z[2]-0.9*y0] 
            x0=(-new_eq[1]+np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
            x1=(-new_eq[1]-np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])   
            ranges=[10**x0, 10**x1]
            return [int(np.ceil(ranges[0]/(block_size_row*block_size_col))),int(np.floor(ranges[1]/(block_size_row*block_size_col)))]
    
    
    def evaluate(self,filename,data,node,benchmark,th,from_formula=False,block=None):  
        titles=['node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','num_elements_uncomplete','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','cache_block','datatype','cost','mflops']
        try:
            d=d_hpx_ref
        except:
            hpx_dir_ref='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/reference_hpx/marvin/master/'         
            d_hpx_ref=create_dict_refernce(hpx_dir_ref)    
        m_data=data[1][node][benchmark]
        i=1
    
        if block is None:
            block='4-1024'
        block_size_row=int(block.split('-')[0])
        block_size_col=int(block.split('-')[-1])
    
        for m in m_data['matrix_sizes']:
            if int(block.split('-')[-1])>m:
                block_size_col=m
            
            c_range=select_chunk_block(data,th,m,node,benchmark,block_size_row,block_size_col,False,from_formula=True)
            
            dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
            for col in titles[2:]:
                dataframe[col] = dataframe[col].astype(float)
            node_selected=dataframe['node']==node
            benchmark_selected=dataframe['benchmark']==benchmark
            th_selected=dataframe['num_threads']==th
            m_selected=dataframe['matrix_size']==m
            df_nb_selected=dataframe[node_selected & benchmark_selected & th_selected & m_selected] 
            df_nb_selected['block_size']=df_nb_selected['block_size_row'].astype(int).astype(str)+'-'+df_nb_selected['block_size_col'].astype(int).astype(str)
            block_sizes=df_nb_selected['block_size'].drop_duplicates().values
            columns=['block_size','chunk_size','mflops']
            df_nb_selected=df_nb_selected[columns]
            
            array=df_nb_selected.values
            plt.figure(i)
            plt.axes([0, 0, 3, 1])
            plt.scatter(array[:,0], array[:,-1],marker='+')
            
            # add some text for labels, title and axes ticks
            
            plt.ylabel('block_sizes')
        
            max_perf=np.max(array[:,-1])
            for x,y,z in zip(array[:,0],array[:,1],array[:,2]):                
                            label = (int(y))                    
                            plt.annotate(label, # this is the text
                                         (x,z), # this is the point to label
                                         textcoords="offset points", # how to position the text
                                         xytext=(20,0), # distance from text to points (x,y)
                                         ha='center') # horizontal alignment can be left, right or center
            
            for chunk_size in range(c_range[0],c_range[1]+1):
                plt.figure(i)
    
    #            print('matrix size: '+str(m),'chunk size: '+str(chunk_size))
                chunk_selected=df_nb_selected['chunk_size']==chunk_size
                block_selected=df_nb_selected['block_size']==block
                try:                    
                    k=d_hpx_ref[node][benchmark][th]['size'].index(m)                                
                    plt.bar('refernce',d_hpx_ref[node][benchmark][th]['mflops'][k],color='green')
                except:
                    pass
    #                print('reference benchmark does not exist')    
                if df_nb_selected[chunk_selected & block_selected]['mflops'].size!=0:
                    prediction=df_nb_selected[chunk_selected & block_selected]['mflops'].values[-1]
                    plt.bar(str(chunk_size),prediction,color='r')             
                    label = block                    
                    plt.annotate(label, # this is the text
                                 (str(chunk_size),prediction), # this is the point to label
                                 textcoords="offset points", # how to position the text
                                 xytext=(0,10), # distance from text to points (x,y)
                                 ha='center') # horizontal alignment can be left, right or center
                    plt.title('matrix size:'+str(int(m))+'  '+str(int(th))+' threads  maximum performance:'+str(max_perf)+'   prediction:'+str(prediction))            
    
                else:
                    plt.bar(str(chunk_size),0,color='r') 
                    plt.title('matrix size:'+str(int(m))+'  '+str(int(th))+' threads  maximum performance:'+str(max_perf)+'  chunk size:'+str(chunk_size)+' no prediction(data for that chunk size was not collected')
    #        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            i=i+1
