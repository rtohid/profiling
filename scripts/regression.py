#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:22:37 2019

@author: shahrzad
"""
import sklearn
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from sklearn.feature_selection import chi2
titles=['node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','num_elements_uncomplete','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','cache_block','datatype','cost','mflops']

dataframe = pandas.read_csv('/home/shahrzad/repos/Blazemark/data/data_perf_all.csv', header=0,index_col=False,dtype=str,names=titles)
for col in titles[2:]:
    dataframe[col] = dataframe[col].astype(float)
#array = dataframe.values
marvin=dataframe['node']=='marvin'
add=dataframe['benchmark']=='dmatdmatadd'

#features=['num_threads','num_elements','grain_size','mflops']
features=titles[2:]
df_selected=dataframe[marvin & add][features]

#array=df_marvin[df_marvin['benchmark']=='dmatdmatadd'].values
#array=array[:,2:].astype(float)
array=df_selected.values
array=array.astype(float)


data_size=np.shape(array)[0]
per = np.random.permutation(data_size)
train_size=int(np.ceil(0.6*data_size))
train_set=array[per[0:train_size],:-1]  
train_labels=array[per[0:train_size],-1]  
test_set=array[per[train_size:],:-1]  
test_labels=array[per[train_size:],-1]  

model = RandomForestRegressor()
model.fit(train_set,train_labels)

#        pp = PdfPages(perf_directory+'/performance_'+benchmark+'_different_blocks-chunk_size_'+str(c)+'.pdf')

# Get the mean absolute error on the validation data
predicted_performances = model.predict(test_set)
MAE = mean_absolute_error(test_labels , predicted_performances)
print('Random forest validation MAE = ', MAE)
print(model.feature_importances_)
print(test_labels[3],predicted_performances[3])
plt.figure(1)
for i in range(len(features)-1):
    plt.bar(i, model.feature_importances_[i],label=features[i])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('model1')
    
features=['matrix_size','num_threads','num_blocks/chunk_size','grain_size','num_elements*chunk_size','cost','mflops']    
df_f=df_selected[features]
df_f.corr()
pandas.plotting.scatter_matrix(df_f)
plt.show()

test = SelectKBest(score_func=f_regression, k=7)

fit=test.fit(train_set,train_labels)
print(fit.scores_)
plt.figure(1)
for i in range(len(features)-1):
    plt.bar(i, fit.scores_[i],label=titles[i])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('model1')
features = test.transform(train_set)


estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names=titles[2:-1], filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=2000'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

#data=np.asarray(all_data)
#data_size=len(all_data)
#per = np.random.permutation(data_size)
#train_size=int(np.ceil(0.6*data_size))
#train_set=data[per[0:train_size],:-1]    
#train_labels=data[per[0:train_size],-1]    
#test_set=data[per[train_size:],:-1]  
#test_labels=data[per[train_size:],-1]    



#pca
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

pca = PCA(n_components=5)
# X is the matrix transposed (n samples on the rows, m features on the columns)
pca.fit(train_set)
pca.components_
pca.explained_variance_ratio_
pca.get_covariance()
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
X_new = pca.transform(train_set)
lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
X=array[:,:-1]
y=array[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

for ix, svr in enumerate(svrs):
    axes[ix].plot(y_test, svr.fit(X_train, y_train).predict(X_test), color=model_color[ix], lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()
#decision tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
X=array[:,:-1]
y=array[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeRegressor()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.mean_absolute_error(y_test, y_pred))

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features[:-1])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())


#neural network
import tensorflow as tf
learning_rate = 0.5
epochs = 10
batch_size = 100


feature_size=len(features)-1
    
 #keras
from keras.layers import Input, Dense
from keras.models import Model
 
# This returns a tensor. Since the input only has one column
inputs = Input(shape=(feature_size,))
 
# a layer instance is callable on a tensor, and returns a tensor
# To the first layer we are feeding inputs
x = Dense(32, activation='relu')(inputs)
# To the next layer we are feeding the result of previous call here it is h
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
 
# Predictions are the result of the neural network. Notice that the predictions are also having one column.
predictions = Dense(1)(x)
 
# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
# Here the loss function is mse - Mean Squared Error because it is a regression problem.
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mse'])

model.fit(train_set, train_labels,  epochs=50, batch_size=100)  # starts training  
y_test = model.predict(test_set)
plt.scatter(test_labels, y_test)

   
#predict grain_size
titles=['node','benchmark','matrix_size','num_threads','num_elements','block_size_row','block_size_col','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','ratio','L1cache','L2cache','L3cache','cache_line','cache_block','datatype','mflops']

dataframe = pandas.read_csv('/home/shahrzad/repos/Blazemark/data/data_perf_all.csv', header=0,index_col=False,dtype=str)
for col in titles[2:]:
    dataframe[col] = dataframe[col].astype(float)
#array = dataframe.values
marvin=dataframe['node']=='marvin'
add=dataframe['benchmark']=='dmatdmatadd'

#features=['num_threads','num_elements','grain_size','mflops']
features=titles[2:]
df_selected=dataframe[marvin & add][features]

#array=df_marvin[df_marvin['benchmark']=='dmatdmatadd'].values
#array=array[:,2:].astype(float)
array=df_selected.values
array=array.astype(float)


#array=df_marvin[df_marvin['benchmark']=='dmatdmatadd'].values
#array=array[:,2:].astype(float)
array=df_selected.values
array=array.astype(float)
indices =np.arange(len(features))
selector = [x for x in range(array.shape[1]) if x != features.index('grain_size')]
per = np.random.permutation(data_size)
train_size=int(np.ceil(0.6*data_size))
train_set=array[per[0:train_size],:]
train_set=train_set[:,selector]  
train_labels=array[per[0:train_size],features.index('grain_size')]  
test_set=array[per[train_size:],:]
test_set=test_set[:,selector]  
test_labels=array[per[train_size:],features.index('grain_size')] 
model = RandomForestRegressor()
model.fit(train_set,train_labels)

#        pp = PdfPages(perf_directory+'/performance_'+benchmark+'_different_blocks-chunk_size_'+str(c)+'.pdf')

# Get the mean absolute error on the validation data
predicted_performances = model.predict(test_set)
MAE = mean_absolute_error(test_labels , predicted_performances)
print('Random forest validation MAE = ', MAE)
print(model.feature_importances_)
print(test_labels[3],predicted_performances[3])
titles=['node','benchmark','matrix_size','num_threads','num_elements','block_rows','block_columns','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads']
plt.figure(1)
for i in range(len(features)-1):
    plt.bar(i, model.feature_importances_[i],label=features[i])
    plt.title('model1')
    
    
    
    
    
    
    
from collections import Counter
matrix_sizes=[a[0] for a in all_data]
Counter(matrix_sizes)

#https://www.tensorflow.org/tutorials/keras/basic_regression
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


features=['mflops','matrix_size','num_threads','num_elements','grain_size']
sns.pairplot(dataframe[features], diag_kind="kde")
plt.show()


train_dataset = df_selected.sample(frac=0.8,random_state=1)
test_dataset = df_selected.drop(train_dataset.index)
train_stats = train_dataset.describe()

train_stats.pop("mflops")
train_stats = train_dataset.transpose()
train_stats

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model
model = build_model()
model.summary()


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
