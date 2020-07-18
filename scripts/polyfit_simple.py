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
from sklearn.metrics import r2_score

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

filename='/home/shahrzad/repos/Blazemark/data/data_perf_all.csv'
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/polynomial'
benchmarks=['dmatdmatadd']
plot=True
error=False
save=False
plot_type='perf_curves'
titles=['runtime','node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','num_elements_uncomplete','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops']
node='marvin'
benchmark='dmatdmatadd'

ranges={}
deg=2
dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[3:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()
node_selected=dataframe['node']==node
df_n_selected=dataframe[node_selected]
benchmark_selected=dataframe['benchmark']==benchmark
num_threads_selected=dataframe['num_threads']<=8
rt_selected=dataframe['runtime']=='hpx'
df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected & rt_selected ]       
matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
matrix_sizes.sort()
thr=df_nb_selected['num_threads'].drop_duplicates().values
thr.sort()

p_ranges={}
m_data={}
m_data['params']={}
for m in matrix_sizes:
    x=[]
    y=[]
    d=[]
    x_test=[]
    y_test=[]
    d_test=[]
    m_data['params'][m]={}
    train_errors=[]
    test_errors=[]
    train_errors_l=[]
    test_errors_l=[]
    R2_tr=[]
    R2_te=[]
    q=1
    np.random.seed(1)
    p_range=[0.,np.inf]
    for th in thr:
        ranges[th]=[]
        m_selected=df_nb_selected['matrix_size']==m
        th_selected=df_nb_selected['num_threads']==th
        
        features=['grain_size','mflops']
        df_selected=df_nb_selected[m_selected & th_selected][features]
        
        array=df_selected.values
        array=array.astype(float)
        real_array=array[:,:-1]                 
        array[:,0]=np.log10(array[:,0])      
        array=remove_duplicates(array)
        
        data_size=np.shape(array)[0]
        if data_size>=8:
        
            per = np.random.permutation(data_size)
            train_size=int(np.ceil(0.6*data_size))
            train_set=array[per[0:train_size],:-1]  
            train_labels=array[per[0:train_size],-1]  
            test_set=array[per[train_size:],:-1]  
            test_labels=array[per[train_size:],-1]  
            test_size=data_size-train_size
            z = np.polyfit(train_set[:,0], train_labels, deg)
            m_data['params'][m][th]=z
            p = np.poly1d(z)
            train_errors.append(np.mean(np.abs(train_labels-p(train_set[:,0]))))

#            train_errors.append(100*np.mean(np.abs(1-p(train_set[:,0])/train_labels)))
            mse=np.mean((train_labels-p(train_set[:,0]))**2)
#            train_errors_l.append(mse/np.var(train_labels))
            R2_tr.append(r2_score(train_labels,p(train_set[:,0])))
            [x.append(k[0]) for k in train_set]
            [y.append(k) for k in train_labels]
            [d.append(th) for k in range(np.shape(train_labels)[0])]
            [x_test.append(k[0]) for k in test_set]
            [y_test.append(k) for k in test_labels]
            [d_test.append(th) for k in range(np.shape(test_labels)[0])]
            
            print(len(x),len(y),len(d))
            if plot_type=='perf_curves' or plot_type=='all': 
                plt.figure(q)
                plt.scatter(train_set[:,0],p(train_set[:,0]),label='fitted',marker='+')
                plt.scatter(train_set[:,0],train_labels,label='true',marker='.')
                test_errors.append(np.mean(np.abs(test_labels-p(test_set[:,0]))))

#                test_errors.append(100*np.mean(np.abs(1-p(test_set[:,0])/test_labels)))
                mse=np.mean((test_labels-p(test_set[:,0]))**2)
                test_errors_l.append(mse/np.var(test_labels))
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.grid(True, 'both')
                plt.xlabel('Grain size')
                plt.ylabel('MFlops')  
                R2_te.append(r2_score(test_labels,p(test_set[:,0])))

                max_perf=np.asarray(-z[1]/(2*z[0]))    
                y0=p(max_perf)
                new_eq=[z[0],z[1],z[2]-0.9*y0] 
                x0=(-new_eq[1]+np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
                x1=(-new_eq[1]-np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])   
    
                ##################
                #plot individual range
#                plt.plot([x0,x1],[p(x0),p(x1)],marker='|',color='red', linewidth=3)
    #            plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_690_'+str(int(th))+'_peak_range_red.png', bbox_inches='tight',dpi=250)
                q=q+1
                ##################
                
                
                ranges[th]=[min(x0,x1),max(x0,x1)]
                if th>1 and min(x0,x1)>p_range[0]:
                    p_range[0]=min(x0,x1)
                if th>1 and max(x0,x1)<p_range[1]:
                    p_range[1]=max(x0,x1) 
                plt.plot(ranges[th],[th,th],marker='+')
                
    #            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.grid(True, 'both')
                plt.xlabel('Grain size')
                plt.ylabel('# cores')  
    ##########################################################################                       
    #plot intersect of ranges                       
    plt.figure(q)
    plt.axvline(p_range[0],color='black')
    plt.axvline(p_range[1],color='black')

#    a=np.linspace(p_range[0],p_range[1],100)
#    #plt.plot([p_range[0],p_range[0]],[thr[0],thr[-1]],color='black')                       
#    #plt.plot([p_range[1],p_range[1]],[thr[0],thr[-1]],color='black')  
#    for i in a:
#        plt.plot([i,i],[thr[0],thr[-1]],color='silver')                       
#        plt.plot([i,i],[thr[0],thr[-1]],color='silver')                       
#    plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_690_peak_range_black.png', bbox_inches='tight',dpi=300)
    ##########################################################################
    p_ranges[m]=p_range
for m in matrix_sizes:
    plt.plot(p_ranges[m],[m,m],marker='+')
    plt.grid(True, 'both')
    plt.xlabel('Grain size')
    plt.ylabel('# cores')  
#    plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_range_all.png', bbox_inches='tight',dpi=300)
#
    p_range=[np.floor(10**p_range[0]),np.floor(10**p_range[1])]

for m in matrix_sizes:
        
    parameters=['a','b','c']
    param_errors=[]
    all_thr=thr
    for i in range(len(m_data['params'][m][th])):
        indices=[j for j in range(len(all_thr)) if j%3!=2]
        plt.figure(i)
        params=np.asarray([m_data['params'][m][th][i] for th in thr])
        plt.scatter(thr,params,label='true')
        z=np.polyfit(thr[indices],params[indices],3)
        p=np.poly1d(z)
    #    plt.scatter(thr[indices],p(thr[indices]),color='orange')
#        plt.scatter(thr,p(thr),label='fitted')
        plt.grid(True, 'both')
        plt.xlabel('# cores')
        plt.ylabel('Parameter '+parameters[i]) 
        not_indices=[j for j in range(len(all_thr)) if j%3==2]
    #    plt.scatter(thr[not_indices],p(thr[not_indices]),label='true',color='blue')
#        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
        param_errors.append([100*np.mean(np.abs(1-p(thr[indices])/params[indices])),100*np.mean(np.abs(1-p(thr[not_indices])/params[not_indices]))])

        plt.savefig('/home/shahrzad/src/Dissertation/Genral_presentation/images/polyfit/fig_'+str(int(m))+'_params_'+str(i)+'before_fit.png', bbox_inches='tight',dpi=300)
    
fig=plt.figure()
ax = fig.add_subplot(111)
width=0.25
rects1 = ax.bar(np.array([0,1,2]),[param_errors[i][0] for i in range(3)], width, color='royalblue',label='training')
rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
plt.xlabel('Parameters')
plt.ylabel('prediction error(%)')
plt.xticks(np.array([0,1,2]))
ax.set_xticklabels(parameters)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


fig=plt.figure()
ax = fig.add_subplot(111)
width=0.25
rects1 = ax.bar(np.arange(1,9),[train_errors[i] for i in range(8)], width, color='royalblue',label='training')
rects2 = ax.bar(np.arange(1,9)+width,[test_errors[i] for i in range(8)], width, color='seagreen',label='test')
plt.xlabel('# Cores')
plt.ylabel('prediction error(MAE)')
plt.xticks(np.array(np.arange(1,9)))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/error_690_mae.png', bbox_inches='tight',dpi=300)

fig=plt.figure()
ax = fig.add_subplot(111)
width=0.25
rects1 = ax.bar(np.arange(1,9),[R2_tr[i] for i in range(8)], width, color='royalblue',label='training')
rects2 = ax.bar(np.arange(1,9)+width,[R2_te[i] for i in range(8)], width, color='seagreen',label='test')
plt.xlabel('# Cores')
plt.ylabel('prediction error(R-squared)')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/error_690_r2.png', bbox_inches='tight',dpi=300)

def polyfit2d(x, y, z, x_p, y_p):
    #m**3*g**2
    ncols = (x_p + 1)*(y_p + 1)
    Q = np.zeros((x.size, ncols))
    ij = itertools.product(range(x_p + 1), range(y_p + 1))
    for k, (i,j) in enumerate(ij):
        Q[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(Q, z)
    return m

def polyval2d(x, z, x_p, z_p, m):
    ij = itertools.product(range(x_p + 1), range(z_p + 1))
    w = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        w += a * x**i * z**j
    return w 

def polyval2d_given_z(z, x_p, z_p, m):  
    ij = itertools.product(range(x_p + 1), range(z_p + 1))
    coef=np.zeros((x_p + 1))
    for a, (i,j) in zip(m, ij):
        coef[x_p-i] += a * z**j
    return np.array(coef)


p_ranges={}
params={}
d_g=2
d_t=4
ranges={}
for th in thr:
    ranges[th]={}
for m in matrix_sizes:
    x=[]
    y=[]
    d=[]
    x_test=[]
    y_test=[]
    d_test=[]
    m_data['params'][m]={}
    train_errors=[]
    test_errors=[]
    q=1
    np.random.seed(1)
    p_range=[0.,np.inf]
    for th in thr:
        ranges[th][m]=[]
        m_selected=df_nb_selected['matrix_size']==m
        th_selected=df_nb_selected['num_threads']==th
        
        features=['grain_size','mflops']
        df_selected=df_nb_selected[m_selected & th_selected][features]
        
        array=df_selected.values
        array=array.astype(float)
        real_array=array[:,:-1]                 
        array[:,0]=np.log10(array[:,0])      
        array=remove_duplicates(array)
        
        data_size=np.shape(array)[0]
        if data_size>=8:
        
            per = np.random.permutation(data_size)
            train_size=int(np.ceil(0.6*data_size))
            train_set=array[per[0:train_size],:-1]  
            train_labels=array[per[0:train_size],-1]  
            test_set=array[per[train_size:],:-1]  
            test_labels=array[per[train_size:],-1]  
            test_size=data_size-train_size
            z = np.polyfit(train_set[:,0], train_labels, deg)
            m_data['params'][m][th]=z
            p = np.poly1d(z)
            train_errors.append(100*np.mean(np.abs(1-p(train_set[:,0])/train_labels)))
            [x.append(k[0]) for k in train_set]
            [y.append(k) for k in train_labels]
            [d.append(th) for k in range(np.shape(train_labels)[0])]
            [x_test.append(k[0]) for k in test_set]
            [y_test.append(k) for k in test_labels]
            [d_test.append(th) for k in range(np.shape(test_labels)[0])]
            
            
    g=np.asarray(x)
    p=np.asarray(y)
    t=np.asarray(d)
    
    g_test=np.asarray(x_test)
    p_test=np.asarray(y_test)
    t_test=np.asarray(d_test)
    
    model=polyfit2d(g, t, p, d_g, d_t)
    pred=polyval2d(g, t, d_g, d_t, model)
    pred_test=polyval2d(g_test, t_test, 2, 4, model)
    params[m]=model
    pred_errors=[]
    pred_errors_corrected=[]
    R2_te=[]
    R2_tr=[]
#    p_range=[0.,np.inf]
    
    for th in thr:
        plt.figure(i)
        indices=[i for i in range(np.shape(t)[0]) if t[i]==th]
        plt.scatter(g[indices],p[indices],color='blue')
        plt.scatter(g[indices],pred[indices],color='green')
        indices_test=[i for i in range(np.shape(t_test)[0]) if t_test[i]==th]
        plt.scatter(g_test[indices_test],p_test[indices_test],color='blue',label='true')
        
        plt.scatter(g_test[indices_test],pred_test[indices_test],color='green',label='fitted')
        z=polyval2d_given_z(th, d_g, d_t, model)
        pd=np.poly1d(z)
    #    plt.scatter(g[indices],p(g[indices]),color='red')
    
        g_i=np.argsort(g[indices])
        g[indices][g_i[0]]
        g_it=np.argsort(g_test[indices_test])
        g_test[indices_test][g_it[0]]
        
        max_perf=np.asarray(-z[1]/(2*z[0]))    
        y0=pd(max_perf)
        new_eq=[z[0],z[1],z[2]-0.90*y0] 
        x0=(-new_eq[1]+np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])
        x1=(-new_eq[1]-np.sqrt(new_eq[1]**2-4*new_eq[0]*new_eq[2]))/(2*new_eq[0])   
        ranges[th][m]=[min(x0,x1),max(x0,x1)]
        
        plt.plot([x0,x1],[pd(x0),pd(x1)],marker='|',color='red', linewidth=5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    #    plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_690_total_'+str(int(th))+'_range.png', bbox_inches='tight',dpi=250)
    #    plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_690_total_'+str(int(th))+'.png', bbox_inches='tight',dpi=250)
        
        train_error_1=np.abs(1-pred[indices]/p[indices])[g_i[0]]
        R2_tr.append(r2_score(p[indices],pred[indices]))
        test_error_1=np.abs(1-pred_test[indices_test]/p_test[indices_test])[g_it[0]]
        R2_te.append(r2_score(p_test[indices_test],pred_test[indices_test]))
#        pred_errors.append([100*np.mean(np.abs(1-pred[indices]/p[indices])),100*np.mean(np.abs(1-pred_test[indices_test]/p_test[indices_test]))])
        pred_errors.append([np.mean(np.abs(p[indices]-pred[indices])),np.mean(np.abs(p_test[indices_test]-pred_test[indices_test]))])

        indices=[j for j in indices if j!=indices[g_i[0]]]
        indices_test=[j for j in indices_test if j!=indices_test[g_it[0]]]
        pred_errors_corrected.append([100*np.mean(np.abs(1-pred[indices]/p[indices])),100*np.mean(np.abs(1-pred_test[indices_test]/p_test[indices_test]))])
    
        i=i+1
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    width=0.25
    rects1 = ax.bar(np.array(thr),[pred_errors[i][0] for i in range(len(thr))], width, color='royalblue',label='training')
    rects2 = ax.bar(np.array(thr)+width,[pred_errors[i][1] for i in range(len(thr))], width, color='seagreen',label='test')
    plt.xlabel('# cores')
    plt.ylabel('prediction error(%)')
    plt.xticks(np.array(thr))
    ax.set_xticklabels([int(th) for th in thr])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_690_total_error.png', bbox_inches='tight',dpi=250)
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    width=0.25
    rects1 = ax.bar(np.array(thr),[R2_tr[i] for i in range(len(thr))], width, color='royalblue',label='training')
    rects2 = ax.bar(np.array(thr)+width,[R2_te[i] for i in range(len(thr))], width, color='seagreen',label='test')
    plt.xlabel('# cores')
    plt.ylabel('prediction error(R-squared)')
    plt.xticks(np.array(thr))
    ax.set_xticklabels([int(th) for th in thr])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    plt.savefig('/home/shahrzad/src/Dissertation/Genral_presentation/images/polyfit/fig_690_total_error_r2.png', bbox_inches='tight',dpi=250)
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    width=0.25
    rects1 = ax.bar(np.array(thr),[pred_errors[i][0] for i in range(len(thr))], width, color='royalblue',label='training')
    rects2 = ax.bar(np.array(thr)+width,[pred_errors[i][1] for i in range(len(thr))], width, color='seagreen',label='test')
    plt.xlabel('# cores')
    plt.ylabel('prediction error(MAE)')
    plt.xticks(np.array(thr))
    ax.set_xticklabels([int(th) for th in thr])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    plt.savefig('/home/shahrzad/src/Dissertation/Genral_presentation/images/polyfit/fig_690_total_error_mae.png', bbox_inches='tight',dpi=250)
    
    
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    width=0.25
    rects1 = ax.bar(np.array(thr),[pred_errors_corrected[i][0] for i in range(len(thr))], width, color='royalblue',label='training')
    rects2 = ax.bar(np.array(thr)+width,[pred_errors_corrected[i][1] for i in range(len(thr))], width, color='seagreen',label='test')
    plt.xlabel('# cores')
    plt.ylabel('prediction error(%)')
    plt.xticks(np.array(thr))
    ax.set_xticklabels([int(th) for th in thr])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_690_total_error_corrected.png', bbox_inches='tight',dpi=250)
    
    i=1
    q=0
    fig=plt.figure(i)
    d=['solid','dashed','dotted']
    dd=['violet','lightgreen','darkgrey','cyan','lightcoral']
    for m in [523.,600.,690.,793.,912.]:
        for th in thr[:-1]:
            plt.plot(ranges[th][m],[th+q/6,th+q/6],marker='|', linewidth=2)
            
            plt.grid(True, 'both')
            plt.xlabel('Grain size')
            plt.ylabel('# cores') 
        plt.plot(ranges[thr[-1]][m],[thr[-1]+q/6,thr[-1]+q/6],marker='|', linewidth=2,label='matrix size: '+str(int(m)))
            
        plt.grid(True, 'both')
        plt.xlabel('Grain size')
        plt.ylabel('# cores')  
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        q=q+1   
#        plt.axvspan(p_range[0], p_range[1], color='silver', alpha=0.5)
#    plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_523-912_peak_range_all.png', bbox_inches='tight',dpi=250)

p_ranges={}
for th in thr:
    p_ranges[th]=[0.,np.inf]
    plt.figure(i)
    for m in matrix_sizes:
        plt.plot(ranges[th][m],[m,m],marker='|', linewidth=3)
        plt.grid(True, 'both')
        plt.xlabel('Grain size')
        plt.ylabel('Matrix size') 
        
        if ranges[th][m][0]>p_ranges[th][0]:
            p_ranges[th][0]=ranges[th][m][0]
        if ranges[th][m][1]<p_ranges[th][1]:
            p_ranges[th][1]=ranges[th][m][1]
    plt.plot(p_ranges[th],[th,th],marker='|', linewidth=3)

#    plt.axvspan(p_range[th][0], p_range[th][1], color='silver', alpha=0.5)

#        plt.plot(p_range[th],[th,th],marker='|', linewidth=3)
    
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True, 'both')
    plt.xlabel('Grain size')
    plt.ylabel('# cores') 

#    plt.ylabel('Matrix size') 
#    plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_'+str(int(th))+'_peak_range_all.png', bbox_inches='tight',dpi=250)

    i=i+1
        
#        p_ranges[m]=p_range
#    [np.floor(10**p_range[0]),np.floor(10**p_range[1])]

for m in matrix_sizes:
    plt.plot(p_ranges[m],[m,m],marker='|', linewidth=3)
        
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True, 'both')
    plt.xlabel('Grain size')
    plt.ylabel('Matrix size')  
    
j=1
for i in range((d_g+1)*(d_t+1)):
    plt.figure(j)
    for deg in [3]:
        plt.figure(j)
        plt.scatter(matrix_sizes,[params[m][i] for m in matrix_sizes],color='blue')
        n=np.polyfit(matrix_sizes,[params[m][i] for m in matrix_sizes],deg)
        p=np.poly1d(n)
        plt.scatter(matrix_sizes,p(matrix_sizes),color='green')
        plt.title('param '+str(i)+' degree '+str(deg))
        j=j+1

    #plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_all_sizes_range_silver.png', bbox_inches='tight',dpi=250)

#    plt.axvspan(p_range[0], p_range[1], color='silver', alpha=0.5)
##############################################################
#checking usl consisitency



##############################################################

#    plt.scatter(np.arange(np.shape(d)[0]),p)
#    plt.scatter(np.arange(np.shape(d)[0]),pred)

#plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_690_params_error.png', bbox_inches='tight',dpi=250)
 
#            q=q+1
    
                
#            plt.savefig('D:\\Dissertation\\images\\polyfit\\fig_'+str(int(th))+'_690_peak.png', bbox_inches='tight',dpi=250)

#            if th==thr[-1]:
#                plt.savefig('D:\\Dissertation\\images\\fig15_690.png', bbox_inches='tight',dpi=250)

#fig=plt.figure()
#ax = fig.add_subplot(111)
#width=0.25
#rects1 = ax.bar(thr,train_errors, width, color='royalblue',label='training')
#rects2 = ax.bar(thr+width,test_errors, width, color='seagreen',label='test')
#plt.xlabel('# cores')
#plt.ylabel('prediction error(%)')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig('D:\\Dissertation\\images\\polyfit\\fig_train_test_690.png', bbox_inches='tight',dpi=250)
#


######################
    #finding range of chunk size

###################
    #evaluate the found chunk sizes
##################
q=1
m=690.
for th in thr[1:]:
    print(th)
    block_size_row=4
    block_size_col=256
    equalshare1=math.ceil(m/block_size_row)
    equalshare2=math.ceil(m/block_size_col)  
    num_blocks=equalshare1*equalshare2
    #p_range=[np.floor(10**p_ranges[m][0]),np.floor(10**p_ranges[m][1])]
    p_range=[np.ceil(10**ranges[th][m][0]),np.floor(10**ranges[th][m][1])]
#    p_range=[np.ceil(10**p_ranges[th][0]),np.floor(10**p_ranges[th][1])]

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
    min_found=False
    max_found=False
    
    
    for i in range(num_blocks):
        if not min_found and sum(num_elements[0:i])>p_range[0]:
            min_c=i
            min_found=True
        if not max_found and sum(num_elements[0:i])>p_range[1]:
            max_c=i-1
            max_found=True
    if not min_found or not max_found:
        print("error")
    print(min_c,max_c)

    th_selected=dataframe['num_threads']==th
    m_selected=dataframe['matrix_size']==m
    df_nbc_selected=dataframe[node_selected & benchmark_selected & th_selected & m_selected] 
    block_r_selected=df_nbc_selected['block_size_row']==block_size_row
    block_c_selected=df_nbc_selected['block_size_col']==block_size_col
    df_nbc_selected=df_nbc_selected[block_r_selected & block_c_selected]
    chunk_sizes=df_nbc_selected['chunk_size'].drop_duplicates().values
    chunk_sizes.sort()
    
    for chunk_size in chunk_sizes:                        
#        print('chunk size: '+str(chunk_size))
        chunk_selected=df_nbc_selected['chunk_size']==chunk_size


#        try:     
#            plt.figure(i)
#            k=d_hpx_ref[evaluate_node][evaluate_benchmark][th]['size'].index(m)                                
#            plt.bar('reference',d_hpx_ref[evaluate_node][evaluate_benchmark][th]['mflops'][k],color='green')
#        except:
#            pass
#                print('reference benchmark does not exist')    
        if df_nbc_selected[chunk_selected]['mflops'].size!=0:
            prediction=df_nbc_selected[chunk_selected]['mflops'].values[-1]
            grain_size=df_nbc_selected[chunk_selected]['grain_size'].values[-1]
            
            plt.figure(q)
            valid_chunk_sizes=[c for c in chunk_sizes if c>=min_c and c<=max_c]
            if len(valid_chunk_sizes)%2==0:
                med=valid_chunk_sizes[int(len(valid_chunk_sizes)/2)]
            else:
                med=valid_chunk_sizes[int((len(valid_chunk_sizes)-1)/2+1)]

            
            if chunk_size>=min_c and chunk_size<=max_c:
#                print(chunk_size)
                if chunk_size==med:
                    plt.scatter(med,prediction,color='red',marker='*',linewidth=4)
                else:
                    plt.scatter(chunk_size,prediction,color='silver')  
                
            else:
                plt.scatter((chunk_size),prediction,color='blue')             
            k=d_hpx_ref[node][benchmark][th]['size'].index(m)  
            v=d_hpx_ref[node][benchmark][th]['mflops'][k]                              
            plt.axhline(y=v,color='green')
            plt.xlabel("Chunk size")           
            plt.ylabel('MFlops')
            plt.xscale('log')
#                        plt.title(benchmark)
            plt.grid(True, 'both')
    #            plt.title('matrix size:'+str(int(m))+'  '+str(th)+' threads predicted range of chunk size:['+str(c_range[0])+','+str(c_range[1])+'] vs ['+str(c_range_exact[0])+','+str(c_range_exact[1])+']')
#    plt.figure(q)
#    plt.savefig('/home/shahrzad/src/Dissertation/images/polyfit/fig_'+str(int(m))+'_chunks_'+str(int(th))+'_'+str(block_size_row)+'-'+str(block_size_col)+'.png', bbox_inches='tight',dpi=300)
    q=q+1           
    


#         if not stop:
#                plt.figure(i)
#                plt.bar(str(chunk_size),prediction,color='r')             
#                label = block                    
#                plt.annotate(label, # this is the text
#                             (str(chunk_size),prediction), # this is the point to label
#                             textcoords="offset points", # how to position the text
#                             xytext=(0,5), # distance from text to points (x,y)
#                             ha='center') # horizontal alignment can be left, right or center
##                plt.title('model created based on data from '+build_benchmark+' on '+build_node+' tested on '+ evaluate_benchmark+' on '+evaluate_node+'\n matrix size:'+str(int(m))+'  '+str(int(th))+' threads   maximum performance:'+str(max_perf)+'   prediction:'+str(prediction))            
#                if chunk_size>median_chunk_size and not stop:
#                    chunk_selected=df_nb_selected['chunk_size']==median_chunk_size
#                    if df_nb_selected[chunk_selected & block_selected]['mflops'].size!=0:
#                        prediction=df_nb_selected[chunk_selected & block_selected]['mflops'].values[-1]
#                        plt.figure(i)
#                        plt.bar('median:'+str(int(median_chunk_size)),prediction,color='r')             
#                        label = block                    
#                        plt.annotate(label, # this is the text
#                                     ('median:'+str(int(median_chunk_size)),prediction), # this is the point to label
#                                     textcoords="offset points", # how to position the text
#                                     xytext=(0,5), # distance from text to points (x,y)
#                                     ha='center') # horizontal alignment can be left, right or center
#                        plt.title('model created based on data from '+build_benchmark+' on '+build_node+' tested on '+ evaluate_benchmark+' on '+evaluate_node+'\n matrix size:'+str(int(m))+'  '+str(int(th))+' threads   maximum performance:'+str(max_perf)+'   prediction:'+str(prediction))                                   
#                        plt.ylabel('mflops')
#                if save:
#                    plt.savefig(pp,format='pdf',bbox_inches='tight')
#                stop=True
#                