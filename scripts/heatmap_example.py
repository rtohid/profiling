import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *

# sphinx_gallery_thumbnail_number = 2
s=[0,10,20,30,40,50,59]
sizes=[30, 60, 103, 200, 300, 455, 600, 1048, 2100, 3193, 7000]
fig, ax = plt.subplots()
thr=np.arange(1,17).tolist()
#sizes=d_openmp[benchmark][th]['size']
hpxmp=np.zeros((len(sizes),len(thr)))
set_cmap('Greys')

for i in range(len(sizes)):
    for j in range(len(thr)):
        hpxmp[i,j]=round(d_hpxmp['dmatdmatadd'][thr[j]]['mflops'][i]/d_openmp['dmatdmatadd'][thr[j]]['mflops'][i],1)
#hpxmp=hpxmp[s,:]
im = ax.imshow(hpxmp)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("ratio", rotation=90, va="bottom")

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
        text = ax.text(j, i, hpxmp[i,j], ha="center", va="center", color="w")

ax.set_title("hpxmp speedup")
fig.tight_layout()
plt.show()



import numpy as np

z=hpxmp[s,:]
c = pcolor(z)
set_cmap('gray')
colorbar()
c = pcolor(z, edgecolors='w', linewidths=1)
axis([1,16])
savefig('plt.png')
show()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:58:29 2019

@author: shahrzad
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=00, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=00, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


mat_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 23, 26, 30, 34, 39, 45, 52, 60, 69, 79,
 90, 103, 118, 135, 155, 176, 200, 230, 264, 300, 345, 396, 455, 523, 600, 690, 793, 912, 1048, 1200, 1380,
 1587, 1825, 2100, 2415, 2777, 3193, 3672, 4222, 4855, 5583, 6420, 7000]
s=[0,10,20,30,40,50,59]

thr=np.arange(1,17).tolist()
sizes=mat_sizes
hpxmp=np.zeros((len(sizes),len(thr)))

for i in range(len(sizes)):
    for j in range(len(thr)):
        hpxmp[i,j]=round(d_hpxmp['dmatdmatadd'][thr[j]]['mflops'][i]/d_openmp['dmatdmatadd'][thr[j]]['mflops'][i],2)
hpxmp=hpxmp[s,:]
        
fig, ax = plt.subplots()

im, cbar = heatmap(hpxmp, sizes, thr, ax=ax,
                   cmap="YlGn", cbarlabel="hpxmp ")
texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()
plt.show()






import seaborn as sns
plt.figure(figsize=(len(thr),len(s)))
pivot_table_openmp = openmp_1_2.pivot('vector_size','num_threads','mfc/s')
plt.xlabel('number of threads', size = 16)
plt.ylabel('matrix_size',size = 7)
plt.title('Mf/s of openMP in dmatdmatadd computation',size = 15)
sns_plot_openmp = sns.heatmap(pivot_table_openmp,annot=True, fmt=".1f",linewidths=.5, square=True,cmap='Blues_r',cbar=True)
sns_plot_openmp.set(xlabel='number of threads', ylabel='vector size (*1000000)')