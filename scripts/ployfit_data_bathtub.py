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

filename='/home/shahrzad/repos/Blazemark/data/data_perf_all.csv'
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/bathtub'
benchmarks=['dmatdmatadd']
plot=True
error=False
save=False
node='marvin'
benchmark='dmatdmatadd'
collect_3d_data=True
build_model=False
plot_type='params_th'
    
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
        t=array[:,1]
        count={}
        for i in range(len(g)):
            if (g[i],t[i]) not in g_dict.keys():
                g_dict[(g[i],t[i])]=p[i]
                count[(g[i],t[i])]=1
            else:
                g_dict[(g[i],t[i])]+=p[i]                
                count[(g[i],t[i])]+=1
        p=np.asarray([g_dict[gd]/count[gd] for gd in g_dict.keys()])
        g=np.asarray([gd[0] for gd in g_dict.keys()])
        t=np.asarray([gd[1] for gd in g_dict.keys()])

        array=np.zeros((np.shape(p)[0],3))
        array[:,0]=g
        array[:,1]=t
        array[:,2]=p
    return array



def find_max_range(filename,benchmarks=None,plot=True,error=False,save=False,perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/polynomial'
,plot_type='perf_curves',collect_3d_data=False,build_model=False):
    titles=['node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','num_elements_uncomplete','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops']

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
    g_params={}
    m_data={}
    threads={}
    all_data={}
    for node in nodes:
        node_selected=dataframe['node']==node
        df_n_selected=dataframe[node_selected]
        ranges[node]={}
        m_data[node]={}
        all_data[node]={}
        g_params[node]={}
        modeled_params[node]={}
        if original_benchmarks is None:
            benchmarks=df_n_selected['benchmark'].drop_duplicates().values
        benchmarks.sort()
        threads[node]={}
        for benchmark in benchmarks:             
            modeled_params[node][benchmark]={}
            g_params[node][benchmark]={}
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
            thr=thr[1:]
            matrix_sizes=matrix_sizes
            m_data[node][benchmark]['threads']=thr
            m_data[node][benchmark]['params']={}
            m_data[node][benchmark]['params_bfit']={}
            m_data[node][benchmark]['params_final_fit']={}

            if save:
                perf_filename=perf_directory+'/'+node+'_'+benchmark+'_'+plot_type+'.pdf'
            else:
                perf_filename=perf_directory+'/'+node+'_'+benchmark+'.pdf'
            pp=''
            if plot and save:
                pp = PdfPages(perf_filename)
            popts=[]
            q=1
            for m in matrix_sizes:
                all_data[node][benchmark][m]={'train':[[],[],[],[]],'test':[[],[],[],[]]}
#            pp = PdfPages(perf_directory+'/'+node+'_'+benchmark+'_'+plot_type+'_bathtub.pdf')
            errors={}
            for th in thr:     
                errors[th]={}
                errors[th]['train']=[]
                errors[th]['test']=[]
                p_range=[0.,np.inf]
                m_data[node][benchmark]['params'][th]={}
                m_data[node][benchmark]['params_bfit'][th]={}
                def my_func_total_h(d,ts,alpha,c,f): 
                    return (alpha*d+ts)/((th-1-np.log(1+(np.exp(th-1)-1)*np.exp(-d))))+d*c+f
                def my_func_total_new(x,a,b,c,x0):
                    return a+b/x+c*(x-x0)**th
                def my_func_total(x,a,b,c,d,x0):
                    return a+b/(th**x)+c*((x-x0))**th+d/(th**(x-x0))

                def my_func_bath(x,b):
                    n=m
                    if m%simdsize!=0:
                        n=m+simdsize-m%simdsize
                    c=np.log10(n*n)/np.log10(th)
                    print(th,c)
                    return (b/(x-th))+(1/(c-x))
                def my_func(x,a,b,alpha):                    
                    return a*x**(b-1)*2**(alpha*x)
                for m in matrix_sizes:
                    simdsize=4.
                    aligned_m=m
                    if m%simdsize!=0:
                        aligned_m=m+simdsize-m%simdsize
#                    benchmark='dmatdmatadd'
                    if benchmark=='dmatdmatadd':                            
                        mflop=(aligned_m)**2                           
                    elif benchmark=='dmatdmatdmatadd':
                        mflop=2*(aligned_m)**2
                    else:
                        mflop=2*(aligned_m)**3        
                    
                    m_selected=df_nb_selected['matrix_size']==m
                    th_selected=df_nb_selected['num_threads']==th
                    block_selected_r=df_nb_selected['block_size_row']==4
                    block_selected_c=df_nb_selected['block_size_col']<=512
                    features=['num_tasks','execution_time','mflops']
                    df_selected=df_nb_selected[m_selected & th_selected][features]

#                    df_selected=df_nb_selected[m_selected & th_selected & block_selected_r & block_selected_c][features]

                    array=df_selected.values
                    array=array.astype(float)
#                    array[:,0]=np.log10(array[:,0])/np.log10(th)   
#                    array[:,0]=np.ceil((aligned_m**2)/array[:,0])
#                    array[:,-1]=mflop/(array[:,-1])   
                    a_s=np.argsort(array[:,0])
                    for ir in range(np.shape(array)[1]):
                        array[:,ir]=array[a_s,ir]

                    array=remove_duplicates(array)
                    
                    data_size=np.shape(array)[0]
                    if data_size>=8:
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
                        train_labels_f=array[per,-1]  
                        train_labels=array[per,1]  

                        test_items=[item for item in np.arange((data_size)) if item not in per]
                        test_set=array[test_items,:-1]  
                        test_labels=array[test_items,-1]  
#                        plt.scatter(array[:,0],array[:,-1],color='blue',label='true')
#                        plt.xscale('log')
#                        plt.xlabel('number of tasks')
#                        plt.ylabel('execution time')
#                        plt.xscale('log')
                        try:
                            popt1, pcov=curve_fit(my_func_total_h,train_set[:,0],train_labels,method='lm')
                            z=my_func_total_h(train_set[:,0],*popt1)  
                            plt.figure(q)
                            plt.scatter(train_set[:,0],train_labels,color='blue',label='true')
                            plt.scatter(train_set[:,0],train_labels_f,color='blue',label='true')
                            plt.scatter(train_set[:,0],z,color='green',label='pred')

                            plt.scatter(train_set[:,0],mflop/z,color='green',label='pred')
                            plt.title('train set    matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
                            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                            
                            train_error=sum([abs(train_labels[w]-z[w])*100/train_labels[w] for w in range(len(z))])/len(z)
                            errors[th]['train'].append(train_error)
                            plt.xscale('log')
                            plt.xlabel('number of tasks')
                            plt.ylabel('execution time')
#                            plt.savefig(pp,format='pdf',bbox_inches='tight')

                            z=my_func_total_h(test_set[:,0],*popt1)  
#                            plt.figure(q+1)

                            plt.scatter(test_set[:,0],test_labels,color='red',label='true')
                            plt.scatter(test_set[:,0],z,color='purple',label='pred')

                            plt.xscale('log')
                            plt.xlabel('number of tasks')
                            plt.ylabel('execution time')
                            plt.title('test set  matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
                            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                            test_error=sum([abs(test_labels[w]-z[w])*100/test_labels[w] for w in range(len(z))])/len(z)
                            errors[th]['test'].append(test_error)
                            q=q+2
#                            plt.savefig(pp,format='pdf',bbox_inches='tight')
                            m_data[node][benchmark]['params'][th][m]=popt1.tolist()
                        except:
                            m_data[node][benchmark]['params'][th][m]=[0]
                            print('no function was fitted for matrix size '+str(int(m))+' with '+str(int(th))+' threads')

                        if build_model or collect_3d_data:
                            for j in range(train_size):
                                all_data[node][benchmark][m]['train'][0].append(train_set[j,0])
                                all_data[node][benchmark][m]['train'][1].append(train_labels[j])
                                all_data[node][benchmark][m]['train'][2].append(float(m))
                                all_data[node][benchmark][m]['train'][3].append(float(th))
                            for j in range(data_size-train_size):
                                all_data[node][benchmark][m]['test'][0].append(test_set[j,0])
                                all_data[node][benchmark][m]['test'][1].append(test_labels[j])
                                all_data[node][benchmark][m]['test'][2].append(float(m))
                                all_data[node][benchmark][m]['test'][3].append(float(th))
                    else:
                        print('no data for matrix size '+str(int(m))+' with '+str(int(th))+' threads')
#            plt.show()
#            pp.close()
                #for a fixed m
                
            for th in thr:
                plt.figure(q)
                plt.axes([0, 0, 2, 1])

                plt.scatter(matrix_sizes,errors[th]['train'],color='green',label='train')
                plt.scatter(matrix_sizes,errors[th]['test'],color='blue',label='test')

                plt.title(str(int(th))+' threads')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                q=q+1
            def uslfit2d(x, y):           
                Q = np.zeros((x.size, 7))
                Q[:,0]=(x-1)*y
                Q[:,1]=x*(x-1)*y
                Q[:,2]=-1
                Q[:,3]=-x*(x-1)
                Q[:,4]=(x-1)*(x**2)*y
                Q[:,5]=-(x-1)
                Q[:,6]=-(1/x)
                m, _, _, _ = np.linalg.lstsq(Q, -y)
                return m            
            
            def uslvalue2d(x, m):           
                y=(m[5]*(x-1)+m[3]*x*(x-1)+m[2]+m[6]/x)/(1+m[0]*(x-1)+m[1]*(x*(x-1))+m[4]*(x-1)*x**2)
                return y

#            def uslfit2d(x, y):           
#                Q = np.zeros((x.size, 8))
#                Q[:,0]=-(x-1)
#                Q[:,1]=-x*(x-1)
#                Q[:,2]=-(x-1)*(x**2)
#                Q[:,3]=y
#                Q[:,4]=(x-1)*y
#                Q[:,5]=(x-1)*x*y
#                Q[:,6]=(x-1)*(x**2)*y
#                Q[:,7]=(x-1)*x**3*y
#                m, _, _, _ = np.linalg.lstsq(Q, -np.ones((x.size,1)))
#                return m            
#            
#            def uslvalue2d(x, m):           
#                y=(1+m[0]*(x-1)+m[1]*x*(x-1)+m[2]*(x-1)*(x**2)+m[7]*(x-1)*x**3)/(m[4]*(x-1)+m[5]*x*(x-1)+m[6]*(x**2)*(x-1))
#                
#                return y


            params={}
            params[node]={}
            
            params[node][benchmark]={}        
            for m in matrix_sizes:
                params[node][benchmark][m]=[[],[],[],[]]
                z0_t=[m_data[node][benchmark]['params'][th][m][0] for th in thr]
                model=uslfit2d(np.asarray(thr),np.asarray(z0_t))
                z0=[[model[6],-model[5],model[5]-model[3],model[3]],[1-model[0],-model[1]+model[0],model[1]-model[4],model[4]]]
                params[node][benchmark][m][0]=model
                z=uslvalue2d(np.asarray(thr),model)            
    #                modeled_params[node][benchmark][th]['z0']=n
                if plot and (plot_type=='params_th' or plot_type=='all'):
                    plt.figure(i)
                    plt.scatter((thr),z0_t,label='real ')
                    plt.scatter(thr,z,label='z[0]')
                    plt.xlabel('threads')
                    plt.ylabel('z[0],z[1],z[2],z[3]')   
                    plt.grid(True, 'both')
                    plt.title(node+' '+benchmark+'  matrix size:'+str(int(m)))
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)   
                    if save:
                        plt.savefig(pp,format='pdf',bbox_inches='tight')

                z1_t=[m_data[node][benchmark]['params'][th][m][1] for th in thr]                
                model=uslfit2d(np.asarray(thr),np.asarray(z1_t))
                params[node][benchmark][m][1]=model
                z1=[[model[6],-model[5],model[5]-model[3],model[3]],[1-model[0],-model[1]+model[0],model[1]-model[4],model[4]]]


                z=uslvalue2d(np.asarray(thr),model)
                if plot and (plot_type=='params_th' or plot_type=='all'):
                    plt.figure(i+1)
                    plt.scatter(thr,z1_t,label ='real ')
                    plt.scatter(thr,z,label='z[1]')
                    plt.xlabel('threads')
                    plt.ylabel('z[0],z[1],z[2],z[3]')   
                    plt.grid(True, 'both')
                    plt.title(node+' '+benchmark+'  matrix size:'+str(int(m)))
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
                    if save:
                        plt.savefig(pp,format='pdf',bbox_inches='tight')
                        
                z2_t=[m_data[node][benchmark]['params'][th][m][2] for th in thr]
                model=uslfit2d(np.asarray(thr),np.asarray(z2_t))
                params[node][benchmark][m][2]=model
                z=uslvalue2d(np.asarray(thr),model)
                z2=[[model[6],-model[5],model[5]-model[3],model[3]],[1-model[0],-model[1]+model[0],model[1]-model[4],model[4]]]

                if plot and (plot_type=='params_th' or plot_type=='all'):
                    plt.figure(i+2)
                    plt.scatter(thr,z2_t,label='real')
                    plt.scatter(thr,z,label='z[2]')
                    plt.xlabel('threads')
                    plt.ylabel('z[0],z[1],z[2],z[3]')   
                    plt.grid(True, 'both')
                    plt.title(node+' '+benchmark+'  matrix size:'+str(int(m)))
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
                    if save:
                        plt.savefig(pp,format='pdf',bbox_inches='tight')        

                z3_t=[m_data[node][benchmark]['params'][th][m][3] for th in thr]
                model=uslfit2d(np.asarray(thr),np.asarray(z3_t))
                params[node][benchmark][m][3]=model
                z=uslvalue2d(np.asarray(thr),model)
                z3=[[model[6],-model[5],model[5]-model[3],model[3]],[1-model[0],-model[1]+model[0],model[1]-model[4],model[4]]]

                if plot and (plot_type=='params_th' or plot_type=='all'):
                    plt.figure(i+3)
                    plt.scatter(thr,z3_t,label='real')
                    plt.scatter(thr,z,label='z[3]')
                    plt.xlabel('threads')
                    plt.ylabel('z[0],z[1],z[2],z[3]')   
                    plt.grid(True, 'both')
                    plt.title(node+' '+benchmark+'  matrix size:'+str(int(m)))
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
                    if save:
                        plt.savefig(pp,format='pdf',bbox_inches='tight')     
                i=i+4
            
            
            plt.scatter(np.arange(4),z0[1])
            plt.scatter(np.arange(4),z1[1])
            plt.scatter(np.arange(4),z2[1])
            plt.scatter(np.arange(4),z3[1])

            def predict_exec_time(m,th,g):
                num_tasks=g #np.ceil(m**2/g)
                params_t=[]
                for i in range(4):
                    z=uslvalue2d(th,params[node][benchmark][m][i])
                    params_t.append(z)
                return my_func_total_h(num_tasks, *params_t)
            
            def predict_exec_time_total(m,th,g):
                num_tasks=g #np.ceil(m**2/g)
                params_t=[]
                for i in range(4):
                    z=uslvalue2d(th,params[node][benchmark][m][i])
                    params_t.append(z)
                return my_func_total_h(num_tasks, *params_t)
                
            i=1
            for m in matrix_sizes:
                h=1
                plt.figure(i)
                plt.axes([0, 0, 3, 1])
                test_data=all_data[node][benchmark][m]['test']
                Es=[]
                Ts=[]
                Ps=[]
                for d in range(len(test_data[0])):
                    th=test_data[3][d]
                    nt=test_data[0][d]
                    t=test_data[1][d]
                    p=predict_exec_time(m,th,nt)
#                    print(d,m, th, nt,t, p,100*abs(1-p/t))
#                    plt.scatter(h,p,label='prediction')
                    Es.append(t-p)
                    Ts.append(t)
                    Ps.append(p)
                plt.hist(Es)
                Es=np.array(Es)
                a=np.argsort(Es)
                new_Es=Es[a[5:-5]]  
                plt.hist(new_Es)

                mu=np.mean(Es)
                cv=np.std(new_Es)/np.mean(new_Es)
                print(m,cv)
                plt.title('matrix size: '+str(int(m)))
                    plt.scatter(d,100*abs(1-p/t),label='true value')                  
                    plt.annotate(str(int(nt)), # this is the text
                                 (d,100*abs(1-p/t)), # this is the point to label
                                 textcoords="offset points", # how to position the text
                                 xytext=(0,10), # distance from text to points (x,y)
                                 ha='center') # horizontal alignment can be left, right or center
             
#                    print(((test_data[2][d]**2)*test_data[1][d]-p)/((test_data[2][d]**2)*test_data[1][d]))
                    plt.title('matrix size '+str(int(test_data[2][d])))
                i=i+1
                
                for nt in [10,100,1000,10000]:
                    es=[]
                    for th in range(2,9):
                        p=predict_exec_time(m,th,nt)
                        es.append((m**2)/p)
                    plt.figure(i)
                    plt.plot(np.arange(2,9),es)
                    plt.title('num tasks: '+str(nt))
                    i=i+1
        perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/bathtub/'
#        pp = PdfPages(perf_directory+'params_th.pdf')
        def m_func(x,a,b):
            change_point=np.sqrt(20*1024*1024/24)
            return np.exp(a*x+b) #+np.log(1+np.exp(n[0]*(x-change_point)))
        
        i=0
        th=8.
        m_sizes=[m for m in matrix_sizes if m>1000]
        exec_times_th=[m_data[node][benchmark]['params'][th][m][i] for m in matrix_sizes if m>1000]
        popt, pcov=curve_fit(m_func,matrix_sizes,[m_data[node][benchmark]['params'][th][m][i] for m in matrix_sizes],method='lm')
        popt, pcov=curve_fit(m_func,m_sizes,exec_times_th,method='lm')

        n=np.polyfit(m_sizes,exec_times_th,1)
        p=np.poly1d(n)
        plt.plot(m_sizes,p(m_sizes),label='fit')
        plt.plot(m_sizes,exec_times_th,label='true')
        m_sizes=[m for m in matrix_sizes if m<1000]
        exec_times_th=[m_data[node][benchmark]['params'][th][m][i] for m in matrix_sizes if m<1000]
        n=np.polyfit((m_sizes),np.log(exec_times_th),1)
        p=np.poly1d(n)
        plt.plot(m_sizes,np.exp(p(m_sizes)),label='fit')
        plt.plot(m_sizes,exec_times_th,label='true')

        for th in thr:
            for i in range(len(m_data[node][benchmark]['params'][th][m])):
                plt.figure(i)
                plt.scatter(matrix_sizes,[m_data[node][benchmark]['params'][th][m][i] for m in matrix_sizes],label='paramater '+str(i)+' '+str(int(th))+'th')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
#        for th in thr:
#        for i in range(len(m_data[node][benchmark]['params'][th][m])):
#            plt.figure(i)
#            plt.savefig(pp,format='pdf',bbox_inches='tight')  
#        plt.show()
#        pp.close()
                for j in range(len(params[node][benchmark][m][i])):
                    plt.figure(i)
                    plt.scatter(matrix_sizes,[params[node][benchmark][m][i][j] for m in matrix_sizes],label='paramater '+str(i)+' '+str(j)+'th')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
                
            for m in matrix_sizes:
                if m!=264.:
                    for i in range(len(m_data[node][benchmark]['params'][th][m])):
                        plt.figure(i+1)
                        z_t=[m_data[node][benchmark]['params'][th][m][i] for th in thr]
                        plt.scatter(thr,z_t,label='matrix size '+str(int(m)))
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
                        plt.ylabel('num threads')
        #                z0_m=[m_data[node][benchmark]['params'][th][m][i] for m in matrix_sizes]
        #                plt.scatter(matrix_sizes,z0_m,label='real')
                        plt.title("parameter "+str(i+1))

#            if plot and (plot_type=='params_th' or plot_type=='all'):  
#                for k_p in ['a','b','alpha']:
#                    m_data[node][benchmark]['params_final_fit'][k_p]=[]
#                    plt.figure(k)
#                    plt.xlabel('threads')
#                    plt.ylabel(k_p)  
#                    plt.grid(True, 'both')
#                    plt.title(node+' '+benchmark)
#                    z0_t=[m_data[node][benchmark]['params_bfit'][th][k_p][0] for th in thr]
##                    z0_t=[(z-min(z0_t))/(max(z0_t)-min(z0_t)) for z in z0_t]
#                    n=np.polyfit(thr,z0_t,4)
#                    m_data[node][benchmark]['params_final_fit'][k_p].append(n)
#                    p0 = np.poly1d(n)
#                    plt.plot(thr,p0(thr),label='z[0]')
#                    plt.scatter(thr,[m_data[node][benchmark]['params_bfit'][th][k_p][0] for th in thr],label='z[0]-real')
#                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#                    plt.plot(np.arange(1,17),p0(np.arange(1,17)),label='z[0]-fit',marker='+')
#                    if save:
#                        plt.savefig(pp,format='pdf',bbox_inches='tight')
#
#                    plt.figure(k+1)
#                    plt.xlabel('threads')
#                    plt.ylabel(k_p)  
#                    plt.grid(True, 'both')
#                    plt.title(node+' '+benchmark)
#                    z1_t=[m_data[node][benchmark]['params_bfit'][th][k_p][1] for th in thr]
#                    n=np.polyfit(thr,z1_t,4)
#                    p1 = np.poly1d(n)
#                    plt.plot(thr,p1(thr),label='z[0]')
#                    plt.plot(np.arange(1,17),p1(np.arange(1,17)),label='z[0]-fit',marker='+')
#                    plt.scatter(thr,[m_data[node][benchmark]['params_bfit'][th][k_p][1] for th in thr],label='z[1]')
#                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    m_data[node][benchmark]['params_final_fit'][k_p].append(n)
#
#                    if save:
#                        plt.savefig(pp,format='pdf',bbox_inches='tight')
#
#                    plt.figure(k+2)
#                    plt.xlabel('threads')
#                    plt.ylabel(k_p)  
#                    plt.grid(True, 'both')
#                    plt.title(node+' '+benchmark)
#                    z2_t=[m_data[node][benchmark]['params_bfit'][th][k_p][2] for th in thr]
#                    n=np.polyfit(thr,z2_t,4)
#                    p2 = np.poly1d(n)
#                    plt.plot(thr,p2(thr),label='z[2]')
#                    plt.plot(np.arange(1,17),p2(np.arange(1,17)),label='z[2]-fit',marker='+')
#                    plt.scatter(thr,[m_data[node][benchmark]['params_bfit'][th][k_p][2] for th in thr],label='z[2]')
#                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    m_data[node][benchmark]['params_final_fit'][k_p].append(n)
#
#                    if save:
#                        plt.savefig(pp,format='pdf',bbox_inches='tight')
#                    
#                    plt.figure(k+3)
#                    plt.xlabel('threads')
#                    plt.ylabel(k_p)  
#                    plt.grid(True, 'both')
#                    plt.title(node+' '+benchmark)
#                    z3_t=[m_data[node][benchmark]['params_bfit'][th][k_p][3] for th in thr]
#                    n=np.polyfit(thr,z3_t,4)
#                    p3 = np.poly1d(n)
#                    plt.plot(thr,p3(thr),label='z[3]')
#                    plt.plot(np.arange(1,17),p3(np.arange(1,17)),label='z[3]-fit',marker='+')
#                    plt.scatter(thr,[m_data[node][benchmark]['params_bfit'][th][k_p][3] for th in thr],label='z[3]')
#                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    m_data[node][benchmark]['params_final_fit'][k_p].append(n)
#
#                    if save:
#                        plt.savefig(pp,format='pdf',bbox_inches='tight')
##                    plt.savefig(pp,format='pdf',bbox_inches='tight')
#                    k=k+4
#                    
                    
            if plot and save:
                plt.show()
                pp.close()
            if build_model:
                models=[]
                for p_num in range(3):
                    tp=[]
                    mp=[]
                    cp=[]
                    for th in thr:
                        for m in matrix_sizes:
                            tp.append(th)
                            mp.append(m)
                            cp.append(m_data[node][benchmark]['params'][th][m][p_num])
                    tp=np.asarray(tp)
                    mp=np.asarray(mp)
                    cp=np.asarray(cp)
                    d_t=4
                    d_m=3
#                    model=polyfit2d(tp,mp,cp,d_t,d_m)
#                    models.append(model)
#                    p=polyval2d(tp,mp,d_t,d_m,model)
                    model=np.polyfit(tp,cp,4)
                    p=np.poly1d(model)
                    plt.axes([0, 0, 3, 1])
                    plt.scatter(np.arange(cp.size),cp,color='red')
                    plt.scatter(np.arange(cp.size),p(tp),color='blue') 
                    models.append(model)
                modeled_params[node][benchmark]['all']=models
    result={'ranges':ranges,'m_data':m_data,'thr':threads,'modeled_params':modeled_params}
    if collect_3d_data or build_model:        
        result['all_data']=all_data
    return result

data=find_max_range('/home/shahrzad/repos/Blazemark/data/data_perf_all.csv',benchmarks=['dmatdmatadd'],save=False,plot=False,plot_type='params_th',collect_3d_data=True,build_model=True)



i=1
for th in thr:
    gt=dict_m_data_t['marvin']['dmatdmatadd'][m][th][0]
    pt=dict_m_data_t['marvin']['dmatdmatadd'][m][th][1]

    popt, pcov=curve_fit(my_func,gt,pt,method='trf')

    pred=my_func(np.asarray(gt),*popt)  
                        
    a_c=modeled_params['marvin']['dmatdmatadd']['all']['a']
    b_c=modeled_params['marvin']['dmatdmatadd']['all']['b']
    alpha_c=modeled_params['marvin']['dmatdmatadd']['all']['alpha']
    
    p=np.poly1d(a_c)
    a=p(th)   
    p=np.poly1d(b_c)
    b=p(np.log(th))             
    p=np.poly1d(alpha_c)
    alpha=p(th)
    pred=my_func(np.asarray(gt),a,b,alpha)
    plt.figure(i)
    plt.plot(gt,np.log(pred),label='pred')
    plt.plot(gt,np.log(pt),label='real')
    plt.xlabel('grain_size')
    plt.ylabel('mflops')
    plt.title('matrix sizes:'+str(int(m))+'  '+str(int(th))+' threads')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.figure(200)
    plt.scatter(th,popt[1],label='popt')
    plt.scatter(th,b,label='b')
    i=i+2
    
    
    
    
def predict(m,th,g):
    d_t=4
    d_m=3
    a=polyval2d(th,m,d_t,d_m,models[0])
    betha=polyval2d(th,m,d_t,d_m,models[1])
    alpha=polyval2d(th,m,d_t,d_m,models[2])
    return my_func(g,a,betha,alpha)

def predict(th,g):
    model=[]
    for k_p in ['a','b','alpha']:
        p=np.poly1d(data['modeled_params'][node][benchmark]['all'][k_p])
        model.append(p(th))  
    return my_func(g,model[0],model[1],model[2])

def predict(m,th,g):
    model=[]
    for k_p in ['a','b','alpha']:
        p_k=[]
        for i in range(4):
            p=np.poly1d(m_data[node][benchmark]['params_final_fit'][k_p][i])
            p_k.append(p(th))
        p_k=np.asarray(p_k)
        p=np.poly1d(p_k)
        model.append(p(m))  
    
    return my_func(g,model[0],model[1],model[2])


def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2,amp3,cen3,sigma3):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen1)**2)/((2*sigma1)**2))) + amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen2)**2)/((2*sigma2)**2))) + amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen3)**2)/((2*sigma3)**2))) 

            
g=np.linspace(0.,6.,30)
plt.plot(g,my_func(g,a,betha,alpha))

dict_m={}
dict_m['marvin']={}
dict_m['marvin']['dmatdmatadd']={}

dict_m_data={}
dict_m_data['marvin']={}
dict_m_data['marvin']['dmatdmatadd']={}


dict_m_data_t={}
dict_m_data_t['marvin']={}
dict_m_data_t['marvin']['dmatdmatadd']={}

popts_m={}
popts_m['marvin']={}
popts_m['marvin']['dmatdmatadd']={}
j=1
for m in matrix_sizes:
    dict_m['marvin']['dmatdmatadd'][m]={}
    dict_m_data['marvin']['dmatdmatadd'][m]=[[],[],[],[]]
    dict_m_data_t['marvin']['dmatdmatadd'][m]={}
    max_p=0.
    for th in thr:
        dict_m_data_t['marvin']['dmatdmatadd'][m][th]=[[],[]]
        gt=data['all_data']['marvin']['dmatdmatadd'][th]['test'][0]
        mt=data['all_data']['marvin']['dmatdmatadd'][th]['test'][1]
        pt=data['all_data']['marvin']['dmatdmatadd'][th]['test'][3]
        tt=[th]*len(gt)
        for i in range(len(gt)):
#            plt.figure(1)
            if mt[i]==m:
                dict_m_data['marvin']['dmatdmatadd'][m][0].append(gt[i])
                dict_m_data['marvin']['dmatdmatadd'][m][1].append(m)
                dict_m_data['marvin']['dmatdmatadd'][m][2].append(th)
                dict_m_data['marvin']['dmatdmatadd'][m][3].append(pt[i])
                dict_m_data_t['marvin']['dmatdmatadd'][m][th][0].append(gt[i])
                dict_m_data_t['marvin']['dmatdmatadd'][m][th][1].append(pt[i])

                if pt[i]>max_p:
                    max_p=pt[i]
        dict_m['marvin']['dmatdmatadd'][m][th]=max_p
    popt, pcov=curve_fit(my_func,thr,[dict_m['marvin']['dmatdmatadd'][m][th] for th in thr],method='trf')
    z=my_func(thr,*popt)  
    plt.figure(j)

    plt.plot(thr,[dict_m['marvin']['dmatdmatadd'][m][th] for th in thr],label=str(int(m)),color='blue') 
    plt.plot(thr,z,label='fit',color='red') 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    j=j+1

    popt, pcov=curve_fit(my_func,thr,[dict_m['marvin']['dmatdmatadd'][m][th] for th in thr],method='trf')
    plt.scatter(m,popt[0])
    plt.ylabel('a')
    plt.xlabel('matrix_sizes')
    
    plt.figure(j+1)

    plt.scatter(m,popt[1])
    plt.ylabel('b')
    plt.xlabel('matrix_sizes')
    
    plt.figure(j+2)

    plt.scatter(m,popt[0])
    plt.ylabel('alpha')
    plt.xlabel('matrix_sizes')
    
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    popts_m['marvin']['dmatdmatadd'][m]=popt
    



def my_func_2(x,y,a_g,b_g,alpha_g,a_t,b_t,alpha_t):                    
    return a_g*x**(b_g-1)*np.exp(alpha_g*x)-a_t*y**(b_t-1)*np.exp(alpha_t*y)


dict_m_data['marvin']['dmatdmatadd'][690.]


a=[popts_m['marvin']['dmatdmatadd'][m][0] for m in matrix_sizes[:-4]]    
z=np.polyfit(matrix_sizes[:-4],a,8)  
p=np.poly1d(z)
plt.scatter(matrix_sizes[:-4],a)
plt.plot(matrix_sizes[:-4],p(matrix_sizes[:-4]))
  

b=[popts_m['marvin']['dmatdmatadd'][m][1] for m in matrix_sizes[:-4]]    
z=np.polyfit(matrix_sizes[:-4],b,3)  
p=np.poly1d(z)
plt.scatter(matrix_sizes[:-4],b)
plt.plot(matrix_sizes[:-4],p(matrix_sizes[:-4]))


alpha=[popts_m['marvin']['dmatdmatadd'][m][2] for m in matrix_sizes[:-4]]    
z=np.polyfit(matrix_sizes[:-4],alpha,2)  
p=np.poly1d(z)
plt.scatter(matrix_sizes[:-4],alpha)
plt.plot(matrix_sizes[:-4],p(matrix_sizes[:-4]))



for m in matrix_sizes:
    

            if mt[i]==690.:     
    #            pp=predict(tt[i],gt[i]) 
    #            plt.scatter(tt[i],pp,color='red')
    #            plt.axes([0, 0, 10, 1])
    #
    #            plt.scatter(i,pp,color='red')
    
                plt.scatter(tt[i],max(data['all_data']['marvin']['dmatdmatadd'][th]['test'][3][i]),color='blue')
            if mt[i]==912.:     
    #            pp=predict(tt[i],gt[i]) 
    #            plt.scatter(tt[i],pp,color='red')
    #            plt.axes([0, 0, 10, 1])
    #
    #            plt.scatter(i,pp,color='red')
    
                plt.scatter(tt[i],data['all_data']['marvin']['dmatdmatadd'][th]['test'][3][i],color='green')


#evaluation
i=1
for th in thr:
    gt=dict_m_data_t['marvin']['dmatdmatadd'][m][th][0]
    pt=dict_m_data_t['marvin']['dmatdmatadd'][m][th][1]

    a_c=data['modeled_params']['marvin']['dmatdmatadd']['all'][0]
    b_c=data['modeled_params']['marvin']['dmatdmatadd']['all'][1]
    alpha_c=data['modeled_params']['marvin']['dmatdmatadd']['all'][2]
    p=np.poly1d(a_c)
    a=p(th)   
    p=np.poly1d(b_c)
    b=p(th)             
    p=np.poly1d(alpha_c)
    alpha=p(th)
    pred=my_func(np.asarray(gt),a,b,alpha)
    plt.figure(i)
    plt.plot(gt,pred,label='pred')
    plt.plot(gt,pt,label='real')
    plt.xlabel('grain_size')
    plt.ylabel('mflops')
    plt.title('matrix sizes:'+str(int(m))+'  '+str(int(th))+' threads')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1
    
    

def get_grain_size(m,chunk_size,block_size_row,block_size_col):
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
    grain_size=sum(num_elements[0:c])    
    return grain_size    
    

def select_chunk_block(data,th,m,node,benchmark,block_size_row,block_size_col=None,plot=False,from_formula=False,return_grain_range=False,exact=False,nonzero=False):
    if block_size_col is None:
        block_size_col=float(m)
    if exact:
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
                
        row_sum=sum(num_elements[0:equalshare2])        

    if not from_formula: 
        ranges=data['ranges']
        return [int(np.ceil(ranges[node][benchmark][th][0]/(block_size_row*block_size_col))),int(np.floor(ranges[node][benchmark][th][1]/(block_size_row*block_size_col)))]
    else:
        g_p=2
        m_p=3
        t_p=3
        if nonzero:
            z=polyval3d_given_yz(m, th, g_p, m_p, t_p, data['modeled_params'][node][benchmark]['nonzero'])
        else:
            z=polyval3d_given_yz(m, th, g_p, m_p, t_p, data['modeled_params'][node][benchmark]['all'])

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
        if exact:
            min_r=int(equalshare2*np.floor(ranges[0]/row_sum))+1
            max_r=int(min_r+equalshare2+1)
            for i in range(min_r,max_r):
                if sum(num_elements[0:i])<ranges[0] and sum(num_elements[0:i+1])>ranges[0]:
                    min_c=i
                    break
                min_c=min_r
            min_r=int(equalshare2*np.floor(ranges[1]/row_sum))+1
            max_r=int(min_r+equalshare2+1)
            for i in range(min_r,max_r):
                if sum(num_elements[0:i])<ranges[1] and sum(num_elements[0:i+1])>ranges[1]:
                    max_c=i-1
                    break            
                max_c=min_r-1
            return [min_c,max_c]

        else:
            if return_grain_range:
                return ranges   
            return [int(np.ceil(ranges[0]/(block_size_row*block_size_col))),int(np.floor(ranges[1]/(block_size_row*block_size_col)))]



def evaluate(filename,data,build_node,build_benchmark,th,evaluate_node=None,evaluate_benchmark=None,from_formula=False,block=None,save=False,perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/polynomial',nonzero=False):  
    titles=['node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','num_elements_uncomplete','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','cache_block','datatype','cost','mflops']
    if evaluate_node is None:
        evaluate_node=build_node
    if evaluate_benchmark is None:
        evaluate_benchmark=build_benchmark
    try:
        d=d_hpx_ref
    except:
        hpx_dir_ref='/home/shahrzad/repos/Blazemark/data/matrix/09-15-2019/reference-chunk_size_fixed/'         
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
        
        c_range=select_chunk_block(data,th,m,build_node,build_benchmark,block_size_row,block_size_col,plot=False,from_formula=True,nonzero=nonzero)
        c_range_exact=select_chunk_block(data,th,m,build_node,build_benchmark,block_size_row,block_size_col,plot=False,from_formula=True,exact=True,nonzero=nonzero)
        print(c_range,c_range_exact)

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
        
        plt.plot([np.log10(l) for l in g_range],[max_perf,max_perf],marker='o',label='from formula',linewidth=3.0)
        plt.plot([np.log10(l) for l in data['ranges'][build_node][build_benchmark][th]],[max_perf,max_perf],marker='x',label='from data-intersection of all matrix sizes',linewidth=3.0)
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

        for chunk_size in range(max(c_range_exact[-1],c_range[-1]),min((c_range_exact[0]-1,c_range[0]-1)),-1):                
            

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
                plt.title('matrix size:'+str(int(m))+'  '+str(th)+' threads predicted range of chunk size:['+str(c_range[0])+','+str(c_range[1])+'] vs ['+str(c_range_exact[0])+','+str(c_range_exact[1])+']')
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
th=4.

data=find_max_range('/home/shahrzad/repos/Blazemark/data/data_perf_all.csv',benchmarks=['dmatdmatadd','dmatdmatdmatadd'],save=False,plot=False,plot_type='params_th',collect_3d_data=True,build_model=True)    
thr=data['thr']  
build_benchmark='dmatdmatadd'
evaluate_benchmark='dmatdmatadd'
build_node='marvin'
evaluate_node='marvin'

for th in thr[evaluate_node][evaluate_benchmark][1:]:
    print(th)
    evaluate('/home/shahrzad/repos/Blazemark/data/data_perf_all.csv',data,build_node,build_benchmark,th,evaluate_node=None,evaluate_benchmark='dmatdmatadd',from_formula=True,block='4-256',save=False,nonzero=False)



def polyval2d(x, z, x_p, z_p, m):
    ij = itertools.product(range(x_p + 1), range(z_p + 1))
    w = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        w += a * x**i * z**j
    return w 
    

g_p=2
m_p=3
t_p=3
x_p=2
y_p=3
z_p=3
#z=polyval3d_given_yz(690., th, g_p, m_p, t_p, data['modeled_params'][node][benchmark]['all'])
#m=polyval3d_given_y(690., x_p, y_p, z_p, data['modeled_params'][node][benchmark]['all'])
#polyval2d(x, z, x_p, z_p, m)
model=polyval3d_given_y(690., x_p, y_p, z_p, data['modeled_params'][node][benchmark]['all'])
num_nonzero_params=np.sum(model>1.e-3)
num_nonzero_params=np.sum(abs(model)>1.e-3)
model[(abs(model)>1.e-3).nonzero()]


model=polyval3d_given_x(3.5, x_p, y_p, z_p, data['modeled_params'][node][benchmark]['all'])
num_nonzero_params=np.sum(abs(model)>1.e-3)
model[(abs(model)>1.e-3).nonzero()]



thr=[1.,2.,4.,6.,8.,10.,12.,14.,16.]
for m in matrix_sizes:
    results=[]
    for th in thr:
        results.append(polyval2d(m, th, y_p, z_p, model))

    plt.plot(thr,results,label='matrix size:'+str(int(m)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('threads')
    plt.ylabel('mflops')

def polyfit3d_find(x, y, z, d, x_p, y_p, z_p):
    nonzeros=data['modeled_params']['marvin']['dmatdmatadd']['nonzero'].nonzero()[0].tolist()
    n=np.shape(data['modeled_params']['marvin']['dmatdmatadd']['nonzero'].nonzero())[1]
    Q = np.zeros((x.size, n))
#    x=x[:n]
#    y=y[:n]
#    z=z[:n]
#    d=d[:n]
    ijl = itertools.product(range(x_p + 1), range(y_p + 1), range(z_p + 1))
    for k, (i,j,l) in enumerate(ijl):
        if k in nonzeros:
            print(i,j,l,k)
            Q[:,nonzeros.index(k)] = x**i * y**j * z**l
    m, _, _, _ = np.linalg.lstsq(Q, d)    
    print(np.dot(Q,m))
    print(d)
    plt.plot(np.arange(x.size),d,marker='o',label='real')
    plt.plot(np.arange(x.size),np.dot(Q,m),marker='+',label='pred')

    return m

inputs=np.zeros((48,4))
dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[2:]:
    dataframe[col] = dataframe[col].astype(float)
node_selected=dataframe['node']==build_node
benchmark_selected=dataframe['benchmark']==build_benchmark
matrix_sizes=data['m_data'][build_node][build_benchmark]['matrix_sizes']
inds=np.random.permutation(len(matrix_sizes))[0:6]


for m in matrix_sizes:
    



x=[]
y=[]
z=[]
d=[]
block_size='4-1024'
for th in [4.,8.]:
    for m in matrix_sizes[inds]:
        for c in range(2,10,2):  
            print(th,m,c)
            th_selected=dataframe['num_threads']==th
            m_selected=dataframe['matrix_size']==m
            c_selected=dataframe['chunk_size']==c
            br_selected=dataframe['block_size_row']==int(block_size.split('-')[0])
            bc_selected=dataframe['block_size_col']==int(block_size.split('-')[1])

            df_selected=dataframe[node_selected & benchmark_selected & th_selected & m_selected & c_selected & br_selected & bc_selected]             
            
            pred=df_selected['mflops'].values[0]
    
            x.append(np.log10(get_grain_size(m,c,int(block_size.split('-')[0]),int(block_size.split('-')[1]))))
            y.append(m)
            z.append(th)
            d.append(pred)
            
x=np.asarray(x)
y=np.asarray(y)
z=np.asarray(z)
d=np.asarray(d)
m=polyfit3d_find(x, y, z, d, 2, 3, 3)
np.shape(polyfit3d_find(x, y, z, d, 2, 3, 3))


nonzeros=data['modeled_params']['marvin']['dmatdmatadd']['nonzero'].nonzero()[0].tolist()
plt.plot(np.arange(len(nonzeros)),data['modeled_params']['marvin']['dmatdmatadd']['all'][nonzeros],marker='+',label='marvin')
plt.plot(np.arange(len(nonzeros)),m,marker='o',label='predicted marvin')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



plt.plot(np.arange(48),data['modeled_params']['trillian']['dmatdmatadd']['all'],marker='o',label='trillian')
plt.plot(np.arange(48),data['modeled_params']['marvin']['dmatdmatadd']['all'],marker='+',label='marvin')
plt.plot(np.arange(48),data['modeled_params']['marvin']['dmatdmatdmatadd']['all'],marker='+',label='3-marvin')

plt.plot(np.arange(48),data['modeled_params']['marvin']['dmatdmatmult']['all'],marker='+',label='mult-marvin')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#plt.plot(np.arange(48),data['modeled_params']['marvin']['dmatdmatdmatadd']['all'],marker='+',label='3-marvin')

#plt.plot(np.arange(48),data['modeled_params']['marvin']['dmatdmatdmatadd']['nonzero']-data['modeled_params']['marvin']['dmatdmatadd']['nonzero'])
data['modeled_params']['trillian']['dmatdmatadd']['nonzero'].nonzero()


#############################################################################
#usl
#############################################################################
def uslfit2d(x, y):           
    Q = np.zeros((x.size, 3))
    Q[:,0]=(x-1)*y
    Q[:,1]=x*(x-1)*y
    Q[:,2]=-x
    m, _, _, _ = np.linalg.lstsq(Q, -y)
    return m

def uslfit2d_given_l(x, y, l):           
    Q = np.zeros((x.size, 2))
    Q[:,0]=(x-1)*y
    Q[:,1]=x*(x-1)*y    
    m, _, _, _ = np.linalg.lstsq(Q, -y+l*x)
    return m

def uslvalue2d(x, m):           
    y=m[2]*x/(1+m[0]*(x-1)+m[1]*(x*(x-1)))
    return y

def uslvalue2d_given_l(x, l, m):           
    y=l*x/(1+m[0]*(x-1)+m[1]*(x*(x-1)))
    return y

def polyfit_negative(x, y, o1,o2):
    ncols = (o2-o1 + 1)
    Q = np.zeros((x.size, ncols))
    for k in range(ncols):
        Q[:,k] = x**(o1+k)
    m, _, _, _ = np.linalg.lstsq(Q, y)
    return m

def polyval_negative(x, m, o1, o2):
    ncols = (o2-o1 + 1)
    w = np.zeros_like(x)
    for k in range(ncols):
        w += m[k]*x**(ncols-o1+k-1)
    return w

def n1_fit(x,m,order):
    ncols=(order+1)
    w = np.zeros_like(x)
    for k in range(ncols):        
        w += m[k]*x**(ncols-k-1)
    value=(4*m[0]*m[2]-m[1]**2)/(4*m[0])
    w=w*(x<(-m[1]/(2*m[0])))+value*(x>=(-m[1]/(2*m[0])))           
    return w
        
def sigmoid(x,k,b,t):
    return (b/(1+10**(-k*(x-t))))


NUM_COLORS = len(data['thr']['marvin']['dmatdmatadd'])
cm = plt.get_cmap('gist_rainbow')

node='marvin'
benchmark='dmatdmatadd'
g_dict=data['g_params'][node][benchmark]
mg=690.
g=3.8993
l=g_dict[m][g]


mg=912.
g_dict=data['g_params'][node][benchmark][mg]
g_models={}
j=1
for (mg,g_dict) in data['g_params'][node][benchmark].items():
    #mg is matrix size
    g_models[mg]=[]
    m0=[]
    m1=[]
    m2=[]
    all_g=[]
    all_g_n_1=[]
    mflops_n_1=[]
    for (g,m_dict) in g_dict.items():
        print(g)
        all_g.append(g)
        for i in range(len(m_dict[0])):
            if m_dict[0][i]==1:
                all_g_n_1.append(g)
                mflops_n_1.append(m_dict[1][i])
    plt.figure(j)    
    plt.scatter(all_g_n_1,mflops_n_1,label='true')
    plt.xlabel('grain_size')
    plt.ylabel('mflops')
    plt.title('mflops with 1 thread with different grain sizes\nmatrix size:'+str(int(mg)))
    m=np.polyfit(all_g_n_1,mflops_n_1,2)
    p=np.poly1d(m)
    plt.scatter(all_g_n_1,p(all_g_n_1),label='pred')
    plt.xlabel('grain_size')
    plt.ylabel('mflops')
    popt, pcov = curve_fit(sigmoid, np.asarray(all_g_n_1), mflops_n_1,method='trf')
    y=sigmoid(np.asarray(all_g_n_1),*popt)
    plt.scatter(np.asarray(all_g_n_1),y,color='purple',label='sigmoid')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    all_g.sort()
    for g in all_g:         
        m_dict=g_dict[g]
        d=np.asarray(m_dict[1])
        z=np.asarray(m_dict[0])
        model=uslfit2d_given_l(z,d,p(g))
        pred=uslvalue2d_given_l(z,p(g),model)
        m0.append(model[0])
        plt.figure(j+1)
        plt.xlabel('grain_size')
        plt.ylabel('m0')
        plt.title('matrix size:'+str(int(mg)))
        plt.scatter(g,model[0])
        m1.append(model[1])
        plt.figure(j+2)
        plt.scatter(g,model[1])
        plt.xlabel('grain_size')
        plt.ylabel('m1')
        plt.title('matrix size:'+str(int(mg)))
        
        
        
    for g in all_g:         
        m_dict=g_dict[g]
        d=np.asarray(m_dict[1])
        z=np.asarray(m_dict[0])
        q=np.arange(1,9).astype(float)

        model=uslfit2d_given_l(z,d,sigmoid(g,*popt))
        pred=uslvalue2d_given_l(q,sigmoid(g,*popt),model)
        plt.scatter(z,d,color='blue')
        plt.scatter(q,pred,color='red')
        
        
        m0.append(model[0])
        plt.figure(j+1)
        plt.xlabel('grain_size')
        plt.ylabel('m0')
        plt.title('matrix size:'+str(int(mg)))
        plt.scatter(g,model[0])
        m1.append(model[1])
        plt.figure(j+2)
        plt.scatter(g,model[1])
        plt.xlabel('grain_size')
        plt.ylabel('m1')
        plt.title('matrix size:'+str(int(mg)))   
    

    g=4.0
    model=uslfit2d_given_l(z,d,sigmoid(g,*popt))
    pred=uslvalue2d_given_l(z,sigmoid(g,*popt),model)
    z=np.arange(1,9).astype(float)
    plt.plot()
    
    model1=np.polyfit(all_g,m0,3)
    p1=np.poly1d(model1)
    plt.figure(j+1)
    plt.xlabel('grain_size')
    plt.ylabel('m0')
    plt.title('matrix size:'+str(int(mg)))
    plt.scatter(all_g,m0,color='blue')
    plt.scatter(all_g,p1(all_g),color='red')
    
    
    model2=np.polyfit(all_g,m1,5)
    p2=np.poly1d(model2)
    plt.figure(j+1)
    plt.xlabel('grain_size')
    plt.ylabel('m1')
    plt.title('matrix size:'+str(int(mg)))
    plt.plot(all_g,(m1),color='blue')
    plt.scatter(all_g,p2(all_g),color='red')
    
    
#    model3=polyfit_negative(np.asarray(all_g),np.asarray(m1),-2,2)
#    p3=polyval_negative(np.asarray(all_g),model3,-2,2)
#    plt.scatter(all_g,p3,color='red')

        j=j+3
    perf_filename=perf_directory+'/usl_'+node+'_'+benchmark+'_'+str(int(mg))+'.pdf'
    pp=PdfPages(perf_filename)
    for (g,m_dict) in g_dict.items():
        if len(m_dict[0])>3:    
            d=np.asarray(m_dict[1])
            z=np.asarray(m_dict[0])

            model=uslfit2d(z,d)
            pred=uslvalue2d(z,model)
            print('error:',np.sqrt(np.sum((pred-d)**2)))
            if np.sqrt(np.sum((pred-d)**2))<20:
                plt.figure(i)
                plt.plot(m_dict[0],m_dict[1],marker='o',label='grain size:'+str(int(g))+' matrix size:'+str(int(mg))+' real')
                plt.xlabel('threads')
                plt.ylabel('mflops')
                m0.append(model[0])
                m1.append(model[1])
                m2.append(model[2])
                print((1-model[0])/model[1])
                all_g.append(g)
                g_models[mg].append(model)
                z=np.arange(1,9).astype(float)
                pred=uslvalue2d(z,model)
                plt.plot(z,pred,marker='+',label='grain size:'+str(int(g))+' matrix size:'+str(int(mg))+' pred')
                plt.xlabel('threads')
                plt.ylabel('mflops')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                i=i+1
                plt.savefig(pp,format='pdf',bbox_inches='tight')
                
    indices=np.argsort(np.asarray(all_g))
    plt.figure(i)
    plt.plot([all_g[i] for i in indices],[m0[i] for i in indices])
    plt.xlabel('grain size')
    plt.ylabel('m0')
    plt.figure(i+1)
    plt.plot([all_g[i] for i in indices],[m1[i] for i in indices])
    plt.xlabel('grain size')
    plt.ylabel('m1')
    plt.figure(i+2)
    plt.plot([all_g[i] for i in indices],[m2[i] for i in indices])
    plt.xlabel('grain size')
    plt.ylabel('m2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(pp,format='pdf',bbox_inches='tight')

    plt.show()
    pp.close()
        
    y=l[0]
    z=l[1]
    d=l[2]
    plt.scatter(y,d)
    plt.xlabel('matrix size')
    plt.ylabel('mflops')



x=[]
y=[]
z=[]
d=[]
for th in data['thr'][node][benchmark]:
    [x.append(i) for i in data['all_data'][node][benchmark][th]['train'][0]]
    [y.append(i) for i in data['all_data'][node][benchmark][th]['train'][1]]
    [z.append(i) for i in data['all_data'][node][benchmark][th]['train'][2]]
    [d.append(i) for i in data['all_data'][node][benchmark][th]['train'][3]]
x=np.asarray(x)
y=np.asarray(y)
z=np.asarray(z)
d=np.asarray(d)

#
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
#node='marvin'
#benchmark='dmatdmatadd'
#
#
#def polyfit2d(x, y, z, x_p, y_p):
#    #m**3*g**2
#    ncols = (x_p + 1)*(y_p + 1)
#    Q = np.zeros((x.size, ncols))
#    ij = itertools.product(range(x_p + 1), range(y_p + 1))
#    for k, (i,j) in enumerate(ij):
#        Q[:,k] = x**i * y**j
#    m, _, _, _ = np.linalg.lstsq(Q, z)
#    return m
#
#def polyval2d(x, y, x_p, y_p, m):
#    ij = itertools.product(range(x_p + 1), range(y_p + 1))
#    w = np.zeros_like(x)
#    for a, (i,j) in zip(m, ij):
#        w += a * x**i * y**j
#    return w
#
#
#
#    
#params_train={}
#params_test={}
#for th in data[2]:
#    x=np.asarray(data[-1][node][benchmark][th]['train'][0])
#    y=np.asarray(data[-1][node][benchmark][th]['train'][1])
#    z=np.asarray(data[-1][node][benchmark][th]['train'][3])
#    x_p=3
#    y_p=2
#    m_train=polyfit2d(x, y, z, x_p, y_p)
#    ncols = (x_p + 1)*(y_p + 1)
#    params_train[th]=m_train
#    x=np.asarray(data[-1][node][benchmark][th]['test'][0])
#    y=np.asarray(data[-1][node][benchmark][th]['test'][1])
#    z=np.asarray(data[-1][node][benchmark][th]['test'][3])
#    m_test=polyfit2d(x, y, z, x_p, y_p)
#    params_test[th]=m_test
#
#
#for j in range(ncols):    
#    plt.figure(j)
#    plt.scatter(data[2],[params_train[t][j] for t in data[2]])
#    m=np.polyfit(data[2],[params_train[t][j] for t in data[2]],3)
#    p=np.poly1d(m)
#    plt.plot(data[2],p(data[2]),color='r')
#    plt.xlabel('num_threads')
#    plt.ylabel('z'+str(j))
#
#
#
#params_train={}
#params_test={}
#x=[]
#y=[]
#z=[]
#d=[]
#for th in data[2]:
#    [x.append(i) for i in data[-1][node][benchmark][th]['train'][0]]
#    [y.append(i) for i in data[-1][node][benchmark][th]['train'][1]]
#    [z.append(i) for i in data[-1][node][benchmark][th]['train'][2]]
#    [d.append(i) for i in data[-1][node][benchmark][th]['train'][3]]
#x=np.asarray(x)
#y=np.asarray(y)
#z=np.asarray(z)
#d=np.asarray(d)
#x_p=2
#y_p=3
#z_p=3
#m_train=polyfit3d(x, y, z, d, x_p, y_p, z_p)
#ncols = (x_p + 1)*(y_p + 1)*(z_p + 1)
#
#per = np.random.permutation(np.shape(x)[0])
#
#plt.figure(1)
#
#num_samples=1000
#a=x[per[0:num_samples]]
#b=y[per[0:num_samples]]
#c=z[per[0:num_samples]]
#e=d[per[0:num_samples]]
#
#prediction=polyval3d(a,b,c,x_p,y_p,z_p,m_train)
#
#plt.plot(np.arange(num_samples),(e-prediction)/e)
#plt.xlabel('sample')
#plt.ylabel('percent error in predicting mflops')
#plt.title('training data')
#
#
#
#x=[]
#y=[]
#z=[]
#d=[]
#for th in data[2]:
#    [x.append(i) for i in data[-2][node][benchmark][th]['test'][0]]
#    [y.append(i) for i in data[-2][node][benchmark][th]['test'][1]]
#    [z.append(i) for i in data[-2][node][benchmark][th]['test'][2]]
#    [d.append(i) for i in data[-2][node][benchmark][th]['test'][3]]
#x=np.asarray(x)
#y=np.asarray(y)
#z=np.asarray(z)
#d=np.asarray(d)
#x_p=2
#y_p=3
#z_p=3
#m_train=polyfit3d(x, y, z, d, x_p, y_p, z_p)
#ncols = (x_p + 1)*(y_p + 1)*(z_p + 1)
#
#per = np.random.permutation(np.shape(x)[0])
#
#plt.figure(1)
#
#num_samples=10000
#a=x[per[0:num_samples]]
#b=y[per[0:num_samples]]
#c=z[per[0:num_samples]]
#e=d[per[0:num_samples]]
#
#prediction=polyval3d(a,b,c,x_p,y_p,z_p,m_train)
#
#prediction=polyval3d(3.,690.,4.,x_p,y_p,z_p,m_train)
#
#
#plt.plot(np.arange(num_samples),(e-prediction)/e)
#plt.xlabel('sample')
#plt.ylabel('percent error in predicting mflops')
#plt.title('test data')
#
#
#def predict_from_model(node,benchmark,model,m,th):
#    x_p=2
#    y_p=3
#    z_p=3
#    z=polyval3d_given_yz(matrix_size, th, x_p, y_p, z_p, m_train)
#    max_perf=np.asarray(-z[1]/(2*z[0]))    
#    y0=p(max_perf)
#    new_eq=[z[0],z[1],z[2]-0.9*y0] 
#    x0=(-new_eq[1]+np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
#    x1=(-new_eq[1]-np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])   
#    ranges=[10**(min(x0,x1)),10**(max(x0,x1))]
#
#a=[1.,1.5,2.,2.5,3.,3.5,4.,4.5]
#n=np.poly1d(z)
#b=n(a)
#plt.plot(a,b)
#import matplotlib.tri as mtri
#from mpl_toolkits import mplot3d
#
#
#m_train= data[-2][node][benchmark]
#for th in [4.,8.]:
#    n=100
#    xx, yy = np.meshgrid(np.linspace(2., 6., n), data[1][node][benchmark]['matrix_sizes'])
#    zz=th*np.ones(np.shape(xx))
#    dd = polyval3d(xx, yy, zz, x_p,y_p,z_p,m_train)
#
#    plot3d(xx,yy,dd)      
#
#def plot3d(xx,yy,zz):
#    X=[]
#    Y=[]
#    Z=[]
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    for i in range(np.shape(xx)[0]):
#        for j in range(np.shape(xx)[1]):
#            X.append(xx[i,j])
#            Y.append(yy[i,j])
#            Z.append(zz[i,j])
#    
#        
#    for angle in range(0,360,10):
#        fig = plt.figure(i)   
#        ax = fig.add_subplot(1,1,1, projection='3d')
#        triang = mtri.Triangulation(X, Y)
#        ax.plot_trisurf(triang, Z, cmap='jet')
#    
#        ax.scatter(X,Y,Z, marker='.', s=10, c="black", alpha=0.5)
#        ax.view_init(elev=10, azim=angle)
#        ax.set_xlabel('Grain size')
#        ax.set_ylabel('Matrix_size')
#        ax.set_zlabel('Mflops')
#        plt.title(benchmark)
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
