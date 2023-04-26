# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:16:40 2022

@author: m161902
"""


import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib as mpl

               
#computes euclidean distance
def computeLoss(y_true,y_pred):
    return tf.reduce_mean(pow(pow(y_true[:,0]-y_pred[:,0],2)+pow(y_true[:,1]-y_pred[:,1],2),0.5),axis=-1)

def computeDistances(y_true,y_pred):
    return pow(pow(y_true[:,0]-y_pred[:,0],2)+pow(y_true[:,1]-y_pred[:,1],2),0.5)

def makeErrorPlots(model,x,y,dataName,distScale,xlim=60,ylim=60,plotPredicted=True,base=None):
    error=model.evaluate(x,y)
    pred=model.predict(x)
    distances=pow(pow(y[:,0]-pred[:,0],2)+pow(y[:,1]-pred[:,1],2),0.5)
    if plotPredicted:
        plt.scatter(pred[:,0],-pred[:,1],c=distances,vmin=distScale[0],vmax=distScale[1])
    else:
        plt.scatter(y[:,0],-y[:,1],c=distances,vmin=distScale[0],vmax=distScale[1])
    plt.xlim(0,xlim)
    plt.ylim(-ylim,0)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel("Distance")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    if base is None:
        plt.title(dataName+" Mean Distance Error: "+str(round(error,2)))
    else:
        plt.title(dataName+" Percent Explained: "+str(round(1-error/base,2)))
    plt.show()
    return

def makeErrorPlotsFromData(pred,y,error,dataName,distScale,xlim=60,ylim=60,plotPredicted=True,savePath=None):
    plt.figure()
    distances=pow(pow(y[:,0]-pred[:,0],2)+pow(y[:,1]-pred[:,1],2),0.5)
    if error is None:
        error=np.mean(distances)
    if plotPredicted:
        plt.scatter(pred[:,0],-pred[:,1],c=distances,vmin=distScale[0],vmax=distScale[1])
    else:
        plt.scatter(y[:,0],-y[:,1],c=distances,vmin=distScale[0],vmax=distScale[1])
    plt.xlim(0,xlim)
    plt.ylim(-ylim,0)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel("Distance")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(dataName+" Mean Distance Error: "+str(round(error,2)))
    if savePath is None:
        plt.show()
    else:
        plt.savefig(savePath)
    return


def dataPreProcess(counts,spots):
    keep=[]
    y=[]
    for i in range(spots.shape[0]):
        query=str(spots.iloc[i,0])+'x'+str(spots.iloc[i,1])
        for j in range(counts.shape[0]):
            if query==counts.index[j]:
                keep.append(j)
                y.append([spots.iloc[i,0],spots.iloc[i,1]])

    counts_reduced=counts.iloc[keep,:]
    X=counts_reduced.astype(dtype='float32')
    y=np.asarray(y,dtype='float32')
    return X,y

def loadGMT(filename, genes_col=1, pathway_col=0):
        data_dict_list = []
        with open(filename) as gmt:

            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    dict = {'group': pathway, 'gene': gene}
                    data_dict_list.append(dict)

        df = pd.DataFrame(data_dict_list)
        # print df.head()

        return df


def indexTrim(dataFrame):
    newIDX=[]
    for idx in dataFrame.index:
        s=idx[1:]
        s=s[:-2]
        newIDX.append(s)
    dataFrame.index=newIDX
    return dataFrame    
   
def computeNodes(nNodes,layers):
    coef=[]
    for n in range(layers):
        coef.append(pow(2,layers-n-1))
    x=int(nNodes//sum(coef))
    return [x*c for c in coef]
 
def isNeighbor(p1,p2):
    diff=(abs(p1[0]-p2[0]),abs(p1[1]-p2[1]))
    if diff==(1,1):
        return True
    elif diff==(0,2):
        return True
    else:
        return False
