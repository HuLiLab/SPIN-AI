# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:48:20 2022

@author: m161902
"""

from SPINAI.utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow import keras
import sys
import os
from SPINAI.explain import DeepExplain


"""
Use virtual environment with tensorflow 1.0
"""

exp=sys.argv[1]

root=exp+'/CV/'
X=pd.read_csv(exp+'/Processed_Data/'+'counts.csv',index_col=0)
y=pd.read_csv(exp+'/Processed_Data/'+'coords.csv',index_col=0)
keep=[a>0.05 for a in X.var()]
genes=X.var()[keep].index
X=X[genes]
foldData=pd.read_csv(root+'/fold_ids.csv',index_col=0).to_numpy()
foldData=foldData.reshape((foldData.shape[0],))
results=pd.read_csv(root+'/cv_results.csv')
optimal=np.argmin(results['CV_Error'])
best_l=results['N Layers'][optimal]
best_lr=results['Learning Rate'][optimal]
modelDir=root+'/'+str(best_l)+'_'+str(best_lr)+'/Models/'
models=os.listdir(modelDir)

#write in the node computation here
nNodes=round(0.5*X.shape[1])
nodesPerLayer=computeNodes(nNodes,int(best_l))

attPath=root+'/Attribution/' 
if not os.path.exists(attPath):
    os.mkdir(attPath)

for model in models:
    print('running')
    modelPath=modelDir+model
    foldIDX=int(model[4])
    sel=[f==foldIDX for f in foldData]
    X_sel=X.loc[sel,:]
    x=X_sel.to_numpy(dtype='float')
    sess=K.get_session()
    with DeepExplain(session=sess) as de:
        x_in=Input(X.shape[1])
        nxt=x_in
        for i in nodesPerLayer:
            nxt=Dense(i,activation="relu", kernel_initializer='he_normal')(nxt)
            nxt=BatchNormalization()(nxt)
        x_out=Dense(2)(nxt)
        dmodel=Model(x_in,x_out)
        optimizer = keras.optimizers.Adam(learning_rate=best_lr)
        dmodel.compile(optimizer=optimizer,loss=computeLoss)
        dmodel.load_weights(modelPath)
        outID=len(dmodel.layers)-1
        d_attributions=de.explain(method="deeplift",
                T=dmodel.layers[outID].output,
                X=dmodel.layers[0].output,
                inputs=dmodel.inputs[0],
                xs=x,
                baseline=None) #default is a 0 baseline; alter to suit needs
        denseDL=pd.DataFrame(d_attributions)
        denseDL.columns=X_sel.columns
        denseDL.index=X_sel.index
        denseDL.to_csv(attPath+'deeplift_attr_'+str(foldIDX)+'.csv')
        
            
        
