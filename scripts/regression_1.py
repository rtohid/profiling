import pandas
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import glob
import matplotlib.tri as mtri
from mpl_toolkits import mplot3d
import math

filename='/home/shahrzad/repos/Blazemark/data/data_perf_all.csv'
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/polynomial'
benchmarks=['dmatdmatadd']
plot=True
error=False
save=False
plot_type='params_th'
def create_dict_refernce(directory):
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

def polyfit3d(x, y, z, d, x_p, y_p, z_p):
    #m**3*g**2
    ncols = (x_p + 1)*(y_p + 1)*(z_p + 1)
    Q = np.zeros((x.size, ncols))
    ijl = itertools.product(range(x_p + 1), range(y_p + 1), range(z_p + 1))
    for k, (i,j,l) in enumerate(ijl):
        Q[:,k] = x**i * y**j * z**l
    m, _, _, _ = np.linalg.lstsq(Q, d)
    return m

def polyval3d(x, y, z, x_p, y_p, z_p, m):
    ijl = itertools.product(range(x_p + 1), range(y_p + 1), range(z_p + 1))
    w = np.zeros_like(x)
    for a, (i,j,l) in zip(m, ijl):
        w += a * x**i * y**j * z**l
    return w

def polyval3d_given_yz(y, z, x_p, y_p, z_p, m):
    ijl = itertools.product(range(x_p + 1), range(y_p + 1), range(z_p + 1))
    coef=[0.]*(x_p+1)
    for a, (i,j,l) in zip(m, ijl):
        coef[x_p-i] += a * y**j * z**l
    return coef


def find_max_range(filename,benchmarks=None,plot=True,error=False,save=False,perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/polynomial'
,plot_type='perf_curves',collect_3d_data=False,build_model=False):
    titles=['node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','num_elements_uncomplete','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','cache_block','datatype','cost','mflops']

    ranges={}
    deg=2
    dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
    for col in titles[2:]:
        dataframe[col] = dataframe[col].astype(float)
    i=1  
    h=1
    k=1
    nodes=dataframe['node'].drop_duplicates().values
    nodes.sort()
    original_benchmarks=benchmarks
    modeled_params={}
    m_data={}
    threads={}
    all_data={}
    for node in nodes:
        node_selected=dataframe['node']==node
        df_n_selected=dataframe[node_selected]
        ranges[node]={}
        m_data[node]={}
        all_data[node]={}
        modeled_params[node]={}
        if original_benchmarks is None:
            benchmarks=df_n_selected['benchmark'].drop_duplicates().values
        benchmarks.sort()
        threads[node]={}
        for benchmark in benchmarks:             
            modeled_params[node][benchmark]={}
            all_data[node][benchmark]={}
            benchmark_selected=dataframe['benchmark']==benchmark
            num_threads_selected=dataframe['num_threads']<=8
            df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected]         
            matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
            matrix_sizes.sort()
            thr=df_nb_selected['num_threads'].drop_duplicates().values
            thr.sort()
            threads[node][benchmark]=thr
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
                                plt.ylabel('z[0],z[1],z[2]')   
                                plt.grid(True, 'both')
                                plt.title(node+' '+benchmark+'  '+str(int(th))+' threads')
                                plt.figure(i)
                                plt.scatter(m,z[1])
                                plt.xlabel('matrix size')
                                plt.ylabel('z[0],z[1],z[2]')  
                                plt.grid(True, 'both')
                                plt.title(node+' '+benchmark+'  '+str(int(th))+' threads')
                                plt.figure(i)
                                plt.scatter(m,z[2])
                                plt.xlabel('matrix size')
                                plt.ylabel('z[0],z[1],z[2]')  
                                plt.grid(True, 'both')
                                plt.title(node+' '+benchmark+'  '+str(int(th))+' threads')
                            if plot_type=='perf_curves' or plot_type=='all':    
                                plt.figure(h)
                                plt.plot([x0,x1],[p(x0),p(x1)],marker='+',label='interval of max performance  matrix size:'+str(m))
                    #            plt.scatter(max_perf,p(max_perf),marker='o',color='r')
                                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                plt.title(node+'  '+benchmark+'  '+str(th)+' threads')
                                plt.grid(True, 'both')
                                plt.xlabel('grain size')
                                plt.ylabel('MFlops')   
                                A=np.argsort(test_set[:,0])
                                g=np.asarray([test_set[a,0] for a in A])
                                mf=np.asarray([test_labels[a] for a in A])
                                plt.figure(h)
                                plt.plot(g, p(g), label='test set fitted with degree '+str(deg)+'  matrix size:'+str(m))
                    #            plt.plot(g,mf,label='test set real')
                                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                plt.title(node+'  '+benchmark+'  '+str(th)+' threads')
                                plt.grid(True, 'both')
                                plt.xlabel('grain size')
                                plt.ylabel('MFlops')   
                                if save and m==matrix_sizes[-1] and (plot_type=='perf_curves' or plot_type=='all'):
                                    plt.savefig(pp,format='pdf',bbox_inches='tight')

                h=h+1

                ranges[node][benchmark][th]=[10**p_range[0], 10**p_range[1]]
                z0_t=[m_data[node][benchmark]['params'][th][m][0] for m in matrix_sizes]
                n=np.polyfit(matrix_sizes,z0_t,3)
                p1 = np.poly1d(n)
#                modeled_params[node][benchmark][th]['z0']=n
                if plot and (plot_type=='params_th' or plot_type=='all'):
                    plt.figure(i)
                    plt.plot(matrix_sizes,p1(matrix_sizes),label='z[0]')
                    plt.xlabel('matrix size')
                    plt.ylabel('z[0],z[1],z[2]')   
                    plt.grid(True, 'both')
                    plt.title(node+' '+benchmark+'  '+str(int(th))+' threads')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    if save:
#                        plt.savefig(pp,format='pdf',bbox_inches='tight')
                z1_t=[m_data[node][benchmark]['params'][th][m][1] for m in matrix_sizes]
                n=np.polyfit(matrix_sizes,z1_t,3)
                p2 = np.poly1d(n)
#                modeled_params[node][benchmark][th]['z1']=n
                if plot and (plot_type=='params_th' or plot_type=='all'):
                    plt.figure(i)
                    plt.plot(matrix_sizes,p2(matrix_sizes),label='z[1]')
                    plt.xlabel('matrix size')
                    plt.ylabel('z[0],z[1],z[2]')   
                    plt.grid(True, 'both')
                    plt.title(node+' '+benchmark+'  '+str(int(th))+' threads')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    if save:
#                        plt.savefig(pp,format='pdf',bbox_inches='tight')
                z2_t=[m_data[node][benchmark]['params'][th][m][2] for m in matrix_sizes]
                n=np.polyfit(matrix_sizes,z2_t,3)
                p3 = np.poly1d(n)
#                modeled_params[node][benchmark][th]['z2']=n
                if plot and (plot_type=='params_th' or plot_type=='all'):
                    plt.figure(i)
                    plt.plot(matrix_sizes,p3(matrix_sizes),label='z[2]')
                    plt.xlabel('matrix size')
                    plt.ylabel('z[0],z[1],z[2]')   
                    plt.grid(True, 'both')
                    plt.title(node+' '+benchmark+'  '+str(int(th))+' threads')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    if save:
                        plt.savefig(pp,format='pdf',bbox_inches='tight')

                i=i+3
                if collect_3d_data:
                    all_data[node][benchmark][th]['train']=[G_train,M_train,T_train,P_train]
                    all_data[node][benchmark][th]['test']=[G_test,M_test,T_test,P_test]

            if plot and (plot_type=='params_m' or plot_type=='all'):
                for m in matrix_sizes:     
                    
                    n=np.polyfit(thr,[m_data[node][benchmark]['params'][t][m][0] for t in thr],2)
                    p1 = np.poly1d(n)
#                    print(m,'z0',n)
                    n=np.polyfit(thr,[m_data[node][benchmark]['params'][t][m][1] for t in thr],2)
                    p2 = np.poly1d(n)
#                    print(m,'z1',n)
                    n=np.polyfit(thr,[m_data[node][benchmark]['params'][t][m][2] for t in thr],2)
                    p3 = np.poly1d(n)
                    
#                    print(m,'z2',n)
                    for th in thr: 
                        plt.figure(k)
                        plt.scatter(th,m_data[node][benchmark]['params'][th][m][0])
                        plt.plot(thr,p1(thr))
                        plt.xlabel('threads')
                        plt.ylabel('z[0]')  
                        plt.grid(True, 'both')
                        plt.title(node+' '+benchmark+'  matrix size:'+str(int(m)))
                        if save and th==thr[-1]:
                            plt.savefig(pp,format='pdf',bbox_inches='tight')

                        plt.figure(k+1)
                        plt.scatter(th,m_data[node][benchmark]['params'][th][m][1])
                        plt.plot(thr,p2(thr))
                        plt.xlabel('threads')
                        plt.ylabel('z[1]')  
                        plt.grid(True, 'both')
                        plt.title(node+' '+benchmark+'  matrix size:'+str(int(m)))
                        if save and th==thr[-1]:
                            plt.savefig(pp,format='pdf',bbox_inches='tight')

                        plt.figure(k+2)
                        plt.scatter(th,m_data[node][benchmark]['params'][th][m][2])
                        plt.plot(thr,p3(thr))
                        plt.xlabel('threads')
                        plt.ylabel('z[2]')  
                        plt.grid(True, 'both')
                        plt.title(node+' '+benchmark+'  matrix size:'+str(int(m)))
                        if save and th==thr[-1]:
                            plt.savefig(pp,format='pdf',bbox_inches='tight')

#                    plt.savefig(pp,format='pdf',bbox_inches='tight')
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
    result={'ranges':ranges,'m_data':m_data,'thr':threads,'modeled_params':modeled_params}
    if collect_3d_data or build_model:        
        result['all_data']=all_data
    return result
    
    
def select_chunk_block(data,th,m,node,benchmark,block_size_row,block_size_col=None,plot=False,from_formula=False,return_grain_range=False):
    if block_size_col is None:
        block_size_col=float(m)

    equalshare1=math.ceil(m/block_size_row)
    equalshare2=math.ceil(m/block_size_col)  
    num_blocks=equalshare1*equalshare2

    num_elements_uncomplete=0
    if block_size_col<m:
        num_elements_uncomplete=(m%block_size_col)*block_size_row
        
    mflop=0
    if benchmark=='dmatdmatadd':                            
        mflop=block_size_row*block_size_col                            
    elif benchmark=='dmatdmatdmatadd':
        mflop=block_size_row*block_size_col*2
    else:
        mflop=block_size_row*block_size_col*(2*m)
        
    num_elements=[mflop]*num_blocks
    if num_elements_uncomplete:
        for j in range(1,equalshare1+1):
            num_elements[j*equalshare2-1]=num_elements_uncomplete
    data_type=8
    grain_size=sum(num_elements[0:c])
                            
    if not from_formula: 
        ranges=data['ranges']
        return [int(np.ceil(ranges[node][benchmark][th][0]/(block_size_row*block_size_col))),int(np.floor(ranges[node][benchmark][th][1]/(block_size_row*block_size_col)))]
    else:
        g_p=2
        m_p=3
        t_p=3
        z=polyval3d_given_yz(m, th, g_p, m_p, t_p, data['modeled_params'][node][benchmark])
#        modeled_params=data[-1]
#        z_m_0=modeled_params[node][benchmark][th]['z0']
#        z_m_1=modeled_params[node][benchmark][th]['z1']
#        z_m_2=modeled_params[node][benchmark][th]['z2']
        #z=[np.poly1d(z_m_0)(m),np.poly1d(z_m_1)(m),np.poly1d(z_m_2)(m)]
        p=np.poly1d(z)
        max_perf=np.asarray(-z[1]/(2*z[0]))    
        y0=p(max_perf)
        new_eq=[z[0],z[1],z[2]-0.95*y0] 
        x0=(-new_eq[1]+np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
        x1=(-new_eq[1]-np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])   
        ranges=[10**min(x0,x1), 10**max(x0,x1)]
        if return_grain_range:
            return ranges        
        return [int(np.ceil(ranges[0]/(block_size_row*block_size_col))),int(np.floor(ranges[1]/(block_size_row*block_size_col)))]



def evaluate(filename,data,build_node,build_benchmark,th,evaluate_node=None,evaluate_benchmark=None,from_formula=False,block=None,save=False,perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/polynomial'):  
    titles=['node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','num_elements_uncomplete','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','cache_block','datatype','cost','mflops']
    if evaluate_node is None:
        evaluate_node=build_node
    if evaluate_benchmark is None:
        evaluate_benchmark=build_benchmark
    try:
        d=d_hpx_ref
    except:
        hpx_dir_ref='/home/shahrzad/repos/Blazemark/data/matrix/06-13-2019/reference_hpx/marvin/master/'         
        d_hpx_ref=create_dict_refernce(hpx_dir_ref)  
    
    m_data=data['m_data'][build_node][build_benchmark]
    

    if block is None:
        block='4-1024'
    block_size_row=int(block.split('-')[0])
    block_size_col=int(block.split('-')[-1])

    try:
        matrix_sizes=d_hpx_ref[evaluate_node][evaluate_benchmark][1]['size']
    except:
        matrix_sizes=m_data['matrix_sizes']
    if save:
        perf_filename=perf_directory+'/'+evaluate_node+'_'+evaluate_benchmark+'_prediction_'+str(int(th))+'_'+block+'.pdf'
        pp = PdfPages(perf_filename)
    i=1
    j=len(matrix_sizes)+1
    kk=2*len(matrix_sizes)+1
    for m in matrix_sizes:
        stop=False
        print(m)
        if int(block.split('-')[-1])>m:
            block_size_col=m
        
        c_range=select_chunk_block(data,th,m,build_node,build_benchmark,block_size_row,block_size_col,plot=False,from_formula=True)
        print(c_range)
        dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
        for col in titles[2:]:
            dataframe[col] = dataframe[col].astype(float)
        node_selected=dataframe['node']==evaluate_node
        benchmark_selected=dataframe['benchmark']==evaluate_benchmark
        th_selected=dataframe['num_threads']==th
        m_selected=dataframe['matrix_size']==m
        df_nb_selected=dataframe[node_selected & benchmark_selected & th_selected & m_selected] 
        df_nb_selected['block_size']=df_nb_selected['block_size_row'].astype(int).astype(str)+'-'+df_nb_selected['block_size_col'].astype(int).astype(str)
        block_sizes=df_nb_selected['block_size'].drop_duplicates().values
        chunk_sizes=df_nb_selected['chunk_size'].drop_duplicates().values
        chunk_sizes.sort()
        columns=['block_size','chunk_size','grain_size','mflops']
        df_nb_selected=df_nb_selected[columns]
        
        array=df_nb_selected.values
        g_range=select_chunk_block(data,th,m,build_node,build_benchmark,block_size_row,block_size_col,plot=False,from_formula=True,return_grain_range=True)
        plt.figure(kk)
        plt.scatter(np.log10((array[:,2]).astype(float)),array[:,-1],marker='+',color='r')
        plt.grid(True,'both')
        max_perf=np.max(array[:,-1])
        
#        plt.plot([np.log10(g_range[0]),np.log10(g_range[0]),np.log10(g_range[1]),np.log10(g_range[1])],[0.,max_perf,max_perf,0.0],marker='o',label='from formula')
#        plt.plot([np.log10(data['ranges'][build_node][build_benchmark][th][0]),np.log10(data['ranges'][build_node][build_benchmark][th][0])
#        ,np.log10(data['ranges'][build_node][build_benchmark][th][1]),np.log10(data['ranges'][build_node][build_benchmark][th][1])],
#        [0.,max_perf,max_perf,0.],marker='x',label='from data-intersection of all matrix sizes')
        
        plt.plot([np.log10(l) for l in g_range],[max_perf,max_perf],marker='o',label='from formula')
        plt.plot([np.log10(l) for l in data['ranges'][build_node][build_benchmark][th]],[max_perf,max_perf],marker='x',label='from data-intersection of all matrix sizes')
        plt.ylabel('mflops')
        plt.xlabel('grain_size')
        plt.title('matrix size:'+str(int(m))+'  '+str(int(th))+'  threads')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if save:
            plt.savefig(pp,format='pdf',bbox_inches='tight')  
        plt.figure(i)
        plt.axes([0, 0, 3, 1])
        plt.scatter(array[:,0], array[:,-1],marker='+')    
        # add some text for labels, title and axes ticks        
        plt.ylabel('mflops')
        for x,y,z in zip(array[:,0],array[:,1],array[:,3]):                
                        label = (int(y))                    
                        plt.annotate(label, # this is the text
                                     (x,z), # this is the point to label
                                     textcoords="offset points", # how to position the text
                                     xytext=(20,0), # distance from text to points (x,y)
                                     ha='center') # horizontal alignment can be left, right or center
                                 
            
        
        median_chunk_size=np.median(c_range)

        
        if median_chunk_size not in chunk_sizes:            
            min_list=[abs(h-median_chunk_size) for h in chunk_sizes]
            median_chunk_size=chunk_sizes[min_list.index(min(min_list))]

        for chunk_size in range(c_range[-1],c_range[0]-1,-1):                
            

#            print('matrix size: '+str(m),'chunk size: '+str(chunk_size))
            chunk_selected=df_nb_selected['chunk_size']==chunk_size
            block_selected=df_nb_selected['block_size']==block
            try:     
                plt.figure(i)
                k=d_hpx_ref[evaluate_node][evaluate_benchmark][th]['size'].index(m)                                
                plt.bar('reference',d_hpx_ref[evaluate_node][evaluate_benchmark][th]['mflops'][k],color='green')
            except:
                pass
#                print('reference benchmark does not exist')    
            if df_nb_selected[chunk_selected & block_selected]['mflops'].size!=0:
                prediction=df_nb_selected[chunk_selected & block_selected]['mflops'].values[-1]
                grain_size=df_nb_selected[chunk_selected & block_selected]['grain_size'].values[-1]
                
                plt.figure(j)
                plt.bar(str(chunk_size),prediction,color='blue')             
                plt.xlabel('chunk size')
                plt.ylabel('mflops')
                plt.title('matrix size:'+str(int(m))+' performance for the predicted range of chunk size:['+str(c_range[0])+','+str(c_range[1])+']')
                if save and chunk_size==c_range[0]:
                   plt.savefig(pp,format='pdf',bbox_inches='tight')   
                if not stop:
                    plt.figure(i)
                    plt.bar(str(chunk_size),prediction,color='r')             
                    label = block                    
                    plt.annotate(label, # this is the text
                                 (str(chunk_size),prediction), # this is the point to label
                                 textcoords="offset points", # how to position the text
                                 xytext=(0,5), # distance from text to points (x,y)
                                 ha='center') # horizontal alignment can be left, right or center
                    plt.title('model created based on data from '+build_benchmark+' on '+build_node+' tested on '+ evaluate_benchmark+' on '+evaluate_node+'\n matrix size:'+str(int(m))+'  '+str(int(th))+' threads   maximum performance:'+str(max_perf)+'   prediction:'+str(prediction))            
                    if chunk_size>median_chunk_size and not stop:
                        chunk_selected=df_nb_selected['chunk_size']==median_chunk_size
                        if df_nb_selected[chunk_selected & block_selected]['mflops'].size!=0:
                            prediction=df_nb_selected[chunk_selected & block_selected]['mflops'].values[-1]
                            plt.figure(i)
                            plt.bar('median:'+str(int(median_chunk_size)),prediction,color='r')             
                            label = block                    
                            plt.annotate(label, # this is the text
                                         ('median:'+str(int(median_chunk_size)),prediction), # this is the point to label
                                         textcoords="offset points", # how to position the text
                                         xytext=(0,5), # distance from text to points (x,y)
                                         ha='center') # horizontal alignment can be left, right or center
                            plt.title('model created based on data from '+build_benchmark+' on '+build_node+' tested on '+ evaluate_benchmark+' on '+evaluate_node+'\n matrix size:'+str(int(m))+'  '+str(int(th))+' threads   maximum performance:'+str(max_perf)+'   prediction:'+str(prediction))                                   
                            plt.ylabel('mflops')
                    if save:
                        plt.savefig(pp,format='pdf',bbox_inches='tight')
                    stop=True
                
#            else:
#                print('Data does not exist for this block_size and chunk_size')
#                plt.bar(str(chunk_size),0,color='r') 
#                plt.title('matrix size:'+str(int(m))+'  '+str(int(th))+' threads  maximum performance:'+str(max_perf)+'  chunk size:'+str(chunk_size)+' no prediction(data for that chunk size was not collected')
#        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        i=i+1
        j=j+1
        kk=kk+1
    if save:
        plt.show()
        pp.close()

        
node='marvin'
benchmark='dmatdmatadd'
m=200.
block='4-256'
th=5.

data=find_max_range('/home/shahrzad/repos/Blazemark/data/data_perf_all.csv',benchmarks=None,save=False,plot=False,plot_type='params_th',collect_3d_data=True,build_model=True)    
thr=data['thr']  
build_benchmark='dmatdmatadd'
evaluate_benchmark='dmatdmatadd'
build_node='marvin'
evaluate_node='marvin'

for th in thr:
    print(th)
    evaluate('/home/shahrzad/repos/Blazemark/data/data_perf_all.csv',data,build_node,build_benchmark,th,evaluate_node=None,evaluate_benchmark='dmatdmatadd',from_formula=True,block='4-512',save=True)





#
########################################################################################
##using plyfit2d and all the matrix sizes
########################################################################################
#
#def find_max_range_3d(filename,benchmarks=None,plot=False,error=False):
#    ranges={}
#
#    dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
#    for col in titles[2:]:
#        dataframe[col] = dataframe[col].astype(float)
#    i=1    
#    nodes=dataframe['node'].drop_duplicates().values
#    nodes.sort()
#    if benchmarks is None:
#        benchmarks=dataframe['benchmark'].drop_duplicates().values
#    benchmarks.sort()
#    
#    m_data={}
#    for node in nodes:
#        ranges[node]={}
#        m_data[node]={}
#        for benchmark in benchmarks: 
#
#            node_selected=dataframe['node']==node
#            benchmark_selected=dataframe['benchmark']==benchmark
#            df_nb_selected=dataframe[node_selected & benchmark_selected] 
#            matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
#            matrix_sizes.sort()
#            thr=df_nb_selected['num_threads'].drop_duplicates().values
#            thr.sort()
#            ranges[node][benchmark]={}
#            m_data[node][benchmark]={'matrix_sizes':[],'threads':[]}
#            m_data[node][benchmark]['matrix_sizes']=matrix_sizes
#            m_data[node][benchmark]['threads']=thr
#            for m in matrix_sizes:            
#                P=[]
#                G=[]
#                T=[]
#                PT=[]
#                GT=[]
#                TT=[] 
#                for th in thr:
#                    m_selected=df_nb_selected['matrix_size']==m
#                    th_selected=df_nb_selected['num_threads']==th
#        
#                    features=['grain_size','mflops']
#                    df_selected=df_nb_selected[m_selected & th_selected][features]
#                
#                    array=df_selected.values
#                    array=array.astype(float)
#                    
#                    array[:,:-1]=np.log10(array[:,:-1])
#                    data_size=np.shape(array)[0]
#                    if data_size>=8:
#                        per = np.random.permutation(data_size)
#                        train_size=int(np.ceil(0.6*data_size))
#                        train_set=array[per[0:train_size],:-1]  
#                        train_labels=array[per[0:train_size],-1]  
#                        test_set=array[per[train_size:],:-1]  
#                        test_labels=array[per[train_size:],-1]  
#                        test_size=data_size-train_size
#
#                        for j in range(train_size):
#                            G.append(train_set[j,0])
#                            P.append(train_labels[j])
#                            T.append(float(th))
#                        for j in range(data_size-train_size):
#                            GT.append(test_set[j,0])
#                            PT.append(test_labels[j])
#                            TT.append(float(th))
#                z = np.polyfit(train_set[:,0], train_labels, deg)
#                p = np.poly1d(z)
#                if error:
#                    s=0
#                    for i in range(test_size):
#                        s=s+(test_labels[i]-p(test_set[i]))**2
#                    s=np.sqrt(s/test_size)
#                    print(test_size,'estimated standard error:'+str(s))
##                        A=np.argsort(train_set[:,0])
##                        g=np.asarray([train_set[a,0] for a in A])
##                        mf=np.asarray([train_labels[a] for a in A])
##                        
##                        plt.plot(g, mf, label='training data')
#                model = polyfit2d(np.asarray(G), np.asarray(T),np.asarray(P))
#                pred_train=polyval2d(np.asarray(G), np.asarray(T), model)
##                plot3d(G,T,pred_train,i)
#                pred_test=polyval2d(np.asarray(GT), np.asarray(TT), model)
##                plot3d(GT,TT,pred_test,i+1)
#                plot3d(GT,TT,PT,i+2)
#                i=i+1
#
#
#            
#                       
#
#def polyfit1d(x, y, order=3):
#    ncols = (order + 1)
#    Q = np.zeros((x.size, ncols))
#    for k in range(order+1):
#        Q[:,k] = x**k
#    m, _, _, _ = np.linalg.lstsq(Q, z)
#    return m
#
#
node='marvin'
benchmark='dmatdmatadd'


def polyfit2d(x, y, z, x_p, y_p):
    #m**3*g**2
    ncols = (x_p + 1)*(y_p + 1)
    Q = np.zeros((x.size, ncols))
    ij = itertools.product(range(x_p + 1), range(y_p + 1))
    for k, (i,j) in enumerate(ij):
        Q[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(Q, z)
    return m

def polyval2d(x, y, x_p, y_p, m):
    ij = itertools.product(range(x_p + 1), range(y_p + 1))
    w = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        w += a * x**i * y**j
    return w



    
params_train={}
params_test={}
for th in data[2]:
    x=np.asarray(data[-1][node][benchmark][th]['train'][0])
    y=np.asarray(data[-1][node][benchmark][th]['train'][1])
    z=np.asarray(data[-1][node][benchmark][th]['train'][3])
    x_p=3
    y_p=2
    m_train=polyfit2d(x, y, z, x_p, y_p)
    ncols = (x_p + 1)*(y_p + 1)
    params_train[th]=m_train
    x=np.asarray(data[-1][node][benchmark][th]['test'][0])
    y=np.asarray(data[-1][node][benchmark][th]['test'][1])
    z=np.asarray(data[-1][node][benchmark][th]['test'][3])
    m_test=polyfit2d(x, y, z, x_p, y_p)
    params_test[th]=m_test


for j in range(ncols):    
    plt.figure(j)
    plt.scatter(data[2],[params_train[t][j] for t in data[2]])
    m=np.polyfit(data[2],[params_train[t][j] for t in data[2]],3)
    p=np.poly1d(m)
    plt.plot(data[2],p(data[2]),color='r')
    plt.xlabel('num_threads')
    plt.ylabel('z'+str(j))



params_train={}
params_test={}
x=[]
y=[]
z=[]
d=[]
for th in data[2]:
    [x.append(i) for i in data[-1][node][benchmark][th]['train'][0]]
    [y.append(i) for i in data[-1][node][benchmark][th]['train'][1]]
    [z.append(i) for i in data[-1][node][benchmark][th]['train'][2]]
    [d.append(i) for i in data[-1][node][benchmark][th]['train'][3]]
x=np.asarray(x)
y=np.asarray(y)
z=np.asarray(z)
d=np.asarray(d)
x_p=2
y_p=3
z_p=3
m_train=polyfit3d(x, y, z, d, x_p, y_p, z_p)
ncols = (x_p + 1)*(y_p + 1)*(z_p + 1)

per = np.random.permutation(np.shape(x)[0])

plt.figure(1)

num_samples=1000
a=x[per[0:num_samples]]
b=y[per[0:num_samples]]
c=z[per[0:num_samples]]
e=d[per[0:num_samples]]

prediction=polyval3d(a,b,c,x_p,y_p,z_p,m_train)

plt.plot(np.arange(num_samples),(e-prediction)/e)
plt.xlabel('sample')
plt.ylabel('percent error in predicting mflops')
plt.title('training data')



x=[]
y=[]
z=[]
d=[]
for th in data[2]:
    [x.append(i) for i in data[-2][node][benchmark][th]['test'][0]]
    [y.append(i) for i in data[-2][node][benchmark][th]['test'][1]]
    [z.append(i) for i in data[-2][node][benchmark][th]['test'][2]]
    [d.append(i) for i in data[-2][node][benchmark][th]['test'][3]]
x=np.asarray(x)
y=np.asarray(y)
z=np.asarray(z)
d=np.asarray(d)
x_p=2
y_p=3
z_p=3
m_train=polyfit3d(x, y, z, d, x_p, y_p, z_p)
ncols = (x_p + 1)*(y_p + 1)*(z_p + 1)

per = np.random.permutation(np.shape(x)[0])

plt.figure(1)

num_samples=10000
a=x[per[0:num_samples]]
b=y[per[0:num_samples]]
c=z[per[0:num_samples]]
e=d[per[0:num_samples]]

prediction=polyval3d(a,b,c,x_p,y_p,z_p,m_train)

prediction=polyval3d(3.,690.,4.,x_p,y_p,z_p,m_train)


plt.plot(np.arange(num_samples),(e-prediction)/e)
plt.xlabel('sample')
plt.ylabel('percent error in predicting mflops')
plt.title('test data')


def predict_from_model(node,benchmark,model,m,th):
    x_p=2
    y_p=3
    z_p=3
    z=polyval3d_given_yz(matrix_size, th, x_p, y_p, z_p, m_train)
    max_perf=np.asarray(-z[1]/(2*z[0]))    
    y0=p(max_perf)
    new_eq=[z[0],z[1],z[2]-0.9*y0] 
    x0=(-new_eq[1]+np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
    x1=(-new_eq[1]-np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])   
    ranges=[10**(min(x0,x1)),10**(max(x0,x1))]

a=[1.,1.5,2.,2.5,3.,3.5,4.,4.5]
n=np.poly1d(z)
b=n(a)
plt.plot(a,b)
import matplotlib.tri as mtri
from mpl_toolkits import mplot3d


m_train= data[-2][node][benchmark]
for th in [4.,8.]:
    n=100
    xx, yy = np.meshgrid(np.linspace(2., 6., n), data[1][node][benchmark]['matrix_sizes'])
    zz=th*np.ones(np.shape(xx))
    dd = polyval3d(xx, yy, zz, x_p,y_p,z_p,m_train)

    plot3d(xx,yy,dd)      

def plot3d(xx,yy,zz):
    X=[]
    Y=[]
    Z=[]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(np.shape(xx)[0]):
        for j in range(np.shape(xx)[1]):
            X.append(xx[i,j])
            Y.append(yy[i,j])
            Z.append(zz[i,j])
    
        
    for angle in range(0,360,10):
        fig = plt.figure(i)   
        ax = fig.add_subplot(1,1,1, projection='3d')
        triang = mtri.Triangulation(X, Y)
        ax.plot_trisurf(triang, Z, cmap='jet')
    
        ax.scatter(X,Y,Z, marker='.', s=10, c="black", alpha=0.5)
        ax.view_init(elev=10, azim=angle)
        ax.set_xlabel('Grain size')
        ax.set_ylabel('Matrix_size')
        ax.set_zlabel('Mflops')
        plt.title(benchmark)
#        filename='/home/shahrzad/repos/Blazemark/results/step_poly_'+str(angle)+'.png'
#        plt.savefig(filename, dpi=96)
#        plt.gca()


#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
##return a range within 10% of max
#def polyval2d_at_t(y, m):
#    p_range=[0.,0.]
#    order = int(np.sqrt(len(m))) - 1
#    ij = itertools.product(range(order+1), range(order+1))
#    z=[0.]*(order+1)
#    for a, (i,j) in zip(m, ij):
#        z[i] += a * y**j
#    p = np.poly1d(z)
#    max_perf=np.asarray(-z[1]/(2*z[0]))    
#    y0=p(max_perf)
#    new_eq=[z[0],z[1],z[2]-0.9*y0] 
#    x0=(-new_eq[1]+np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
#    x1=(-new_eq[1]-np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
#    if x0>p_range[0]:
#        p_range[0]=x0
#    if x1<p_range[1]:
#        p_range[1]=x1  
#    return p_range
## Fit a 3rd order, 2d polynomial
#m = polyfit2d(x,y,z)
##m = polyfit1d(x,z)
##m = polyfit1d(y,z)
#
## Evaluate it on a grid...
#n=100
#xx, yy = np.meshgrid((np.array([1.,2.,4.,8.,10.,12.,16.])), 
#                     np.linspace(2., 6., n))
#zz = polyval2d(xx, yy, m)
#
## Plot
#plt.imshow(zz, extent=(x.min(), y.max(), x.max(), y.min()))
#plt.scatter(x, y, c=z)
#plt.show()        
#
#
#
#
#
#X=[]
#Y=[]
#Z=[]
#

#    
#    
#    
#    
#
#
#
#
#
#
#
##    plt.figure(i+1)
##    plt.plot(xp,a(xp), label=str(deg))
##    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        
##from sklearn.decomposition import PCA
##from sklearn.svm import SVR
##from sklearn.model_selection import train_test_split
##
##model = RandomForestRegressor()
##model.fit(train_set,train_labels)
##
###        pp = PdfPages(perf_directory+'/performance_'+benchmark+'_different_blocks-chunk_size_'+str(c)+'.pdf')
##
### Get the mean absolute error on the validation data
##predicted_performances = model.predict(test_set)
##MAE = mean_absolute_error(test_labels , predicted_performances)
##print('Random forest validation MAE = ', MAE)
##print(model.feature_importances_)
##print(test_labels[3],predicted_performances[3])
##plt.figure(1)
##for i in range(len(features)-1):
##    plt.bar(i, model.feature_importances_[i],label=features[i])
##    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##    plt.title('model1')
##    
##
##titles.index('grain_size')
##
##
##
##
##
###pca = PCA(n_components=5)
#### X is the matrix transposed (n samples on the rows, m features on the columns)
###pca.fit(train_set)
###pca.components_
###pca.explained_variance_ratio_
###pca.get_covariance()
###X_new = pca.transform(train_set)
###Y_new = pca.transform(test_set)
##clf=SVR()
##
##clf = SVR(kernel='rbf', C=100, gamma='scale', epsilon=.1)
###clf = SVR(kernel='linear', C=100, gamma='auto')
###clf = SVR(kernel='poly', C=100, gamma='auto', degree=2, epsilon=.1,coef0=1)
##
##A=np.argsort(train_set[:,0])
##g=[train_set[a,0] for a in A]
##mf=[train_labels[a] for a in A]
##plt.figure(1)
##plt.plot(g,mf, label='train',marker='*')
##
##clf.fit(train_set,train_labels)
##predicted_performances = clf.predict(test_set)
##MAE = mean_absolute_error(test_labels , predicted_performances)
##
##A=np.argsort(test_set[:,0])
##g=[test_set[a,0] for a in A]
##mf=[test_labels[a] for a in A]
##p=[predicted_performances[a] for a in A]
##plt.figure(1)
##plt.plot(g,mf, label='true',marker='+')
##plt.plot(g,p, label='predicted',marker='o')
##plt.legend()
##
##from sklearn.kernel_ridge import KernelRidge
##clf = KernelRidge(alpha=1.0)
##clf.fit(train_set,train_labels)
##predicted_performances = clf.predict(test_set)
##MAE = mean_absolute_error(test_labels , predicted_performances)
##
##A=np.argsort(test_set[:,0])
##g=[test_set[a,0] for a in A]
##mf=[test_labels[a] for a in A]
##p=[predicted_performances[a] for a in A]
##plt.figure(1)
##plt.plot(g,mf, label='true',marker='+')
##plt.plot(g,p, label='predicted',marker='o')
##plt.legend()
#
#
