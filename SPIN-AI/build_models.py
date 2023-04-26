# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:35:41 2022

@author: m161902
"""
from SpinAI.utils import *
from tensorflow.keras.models import Model
from tensorflow import keras
import sys
import os
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gc


#Function that handles model building and training, return the best model
def modelConstruct(X,y,nodesPerLayer,learningRate,modelPath,Xval,yval,Xtest,ytest):
    x_in=Input(X.shape[1])
    nxt=x_in
    for i in nodesPerLayer:
        nxt=Dense(i,activation="relu", kernel_initializer='he_normal')(nxt)
        nxt=BatchNormalization()(nxt)
    x_out=Dense(2)(nxt)
    dmodel=Model(x_in,x_out)
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    dmodel.compile(optimizer=optimizer,loss=computeLoss)
    dmodel.summary()
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = keras.callbacks.ModelCheckpoint(modelPath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    mhistory=dmodel.fit(X,y,epochs=100,batch_size=32,callbacks=[es,mc],validation_data=(Xval,yval))
    dmodel.load_weights(modelPath)
    print(dmodel.evaluate(Xtest,ytest))
    pred=dmodel.predict(Xtest)
    pred=np.hstack((pred,ytest))
    return pred

def foldExecute(X,y,cvDir,folds,nodesPerLayer,lr):
    modelDir=cvDir+'/Models/'

    if not os.path.exists(modelDir):
        os.mkdir(modelDir)

    predictions=[]
    for testID in range(10):
        #testSamples=[a==testID for a in folds]
        #trainSamples=[a!=testID for a in folds]
        valID=testID+1
        if valID==10:
            valID=0
        testSamples=[a==testID for a in folds]
        valSamples=[a==valID for a in folds]
        trainSamples=[not(testSamples[a] or valSamples[a]) for a in range(len(folds))]

        X_train=X[trainSamples,:]
        X_test=X[testSamples,:]
        y_train=y[trainSamples,:]
        y_test=y[testSamples,:]
        X_val=X[valSamples,:]
        y_val=y[valSamples,:]

        modelPath=modelDir+'fold'+str(testID)+'.h5'
        p=modelConstruct(X_train,y_train,nodesPerLayer,lr,modelPath,X_val,y_val,X_test,y_test)
        predictions.append(p)

    pred=np.vstack(predictions)
    pred=np.hstack((pred,folds.reshape((len(folds),1))))
    pd.DataFrame(pred).to_csv(cvDir+'predictions.csv')
    errors=computeDistances(pred[:,0:2],pred[:,2:4])
    cvError=np.mean(errors)
    return(cvError)



#Function that takes data paths and out directory
def executeModeling(x_path,y_path,outDir):
    #Read in data
    X=pd.read_csv(x_path,index_col=0)
    y=pd.read_csv(y_path,index_col=0).to_numpy()
    keep=[a>0.05 for a in X.var()]
    genes=X.var()[keep].index
    X=X[genes]
    #Read in data
    #scaler=MinMaxScaler()
    #X=scaler.fit_transform(X)
    X=X.to_numpy()
    print(X.shape)
    print(y.shape)
    #Construct Folds
    sample_per_fold=(X.shape[0]//10)
    fold_ids=[np.repeat(i,sample_per_fold) for i in range(10)]
    fold_ids.append(np.repeat(9,X.shape[0]-sample_per_fold*10))
    fold_ids=np.array([item for sublist in fold_ids for item in sublist])
    np.random.shuffle(fold_ids)
    pd.DataFrame(fold_ids).to_csv(outDir+'fold_ids.csv')

    #Set Up Parameter Tuning
    learningRate=[0.1,0.01,0.001]
    nLayers=[1,3,5]
    paramCombos=list(itertools.product(nLayers,learningRate))
    paramCombos=pd.DataFrame(paramCombos,columns=["N Layers","Learning Rate"])
    nNodes=round(0.5*X.shape[1])
    print(paramCombos)
    print('Hello')
    #Per Parameter Set
    cvErrors=[]
    for index,row in paramCombos.iterrows():
        print(row)
        l=row[0]
        lr=row[1]
        nodesPerLayer=computeNodes(nNodes,int(l))
        paramDir=outDir+str(int(l))+'_'+str(lr)+'/'
        if not os.path.exists(paramDir):
            os.mkdir(paramDir)

        cvErrors.append(foldExecute(X,y,paramDir,fold_ids,nodesPerLayer,lr))
        gc.collect()

    #Find the best parameter set
    paramCombos['CV_Error']=cvErrors
    paramCombos.to_csv(outDir+'cv_results.csv')
    optimal=np.argmin(paramCombos['CV_Error'])
 
    #Visualize Predictions for that set
    best_l=paramCombos['N Layers'][optimal]
    best_lr=paramCombos['Learning Rate'][optimal]
    bestPred=pd.read_csv(outDir+str(best_l)+'_'+str(best_lr)+'/predictions.csv',index_col=0).to_numpy()
    makeErrorPlotsFromData(bestPred[:,0:2], bestPred[:,2:4], None, "Slide 1", (0,40),plotPredicted=False,savePath=outDir+'pred.png')
    return

#Main runs the function in a loop

if __name__=='__main__':
    #collect arguments
    args=sys.argv

    patientName=args[1]

    #Create General Directory
    dataDir=patientName+'/Processed_Data/'
    rootDir=patientName+'/CV/'

    if not os.path.exists(rootDir):
        os.mkdir(rootDir)

    outDir=rootDir+'/Slide_1/'

    if not os.path.exists(outDir):
        os.mkdir(outDir)

    executeModeling(dataDir+'X_train.csv',dataDir+'y_train.csv',outDir)

    outDir=rootDir+'/Slide_2/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    executeModeling(dataDir+'X_val.csv',dataDir+'y_val.csv',outDir)

    outDir=rootDir+'/Slide_3/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    executeModeling(dataDir+'X_test.csv',dataDir+'y_test.csv',outDir)    


