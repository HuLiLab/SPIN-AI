from utils import *
import sys
import os
import pandas as pd
import numpy as np
import re
#use tensorflow 2.0 to save myself a headache

#collect arguments
args=sys.argv

patientName=args[1]
countFile=args[2]
rawPath=args[3]
outDir=args[4]

#Create Patient Directory
dataDir=outDir+'/Processed_Data/'
if not os.path.exists(dataDir):
    os.mkdir(dataDir)

#import data
X=pd.read_csv(countFile,index_col=0,delimiter=",")


#data load processing
X=X.transpose()
X_train=X.loc[["_1" in name for name in X.index],:]
X_val=X.loc[["_2" in name for name in X.index],:]
X_test=X.loc[["_3" in name for name in X.index],:]

#get spot data
rawFiles=os.listdir(rawPath)

spotFiles=[]
for file in rawFiles:
    if re.search("_spot_data-selection-"+patientName,file):
        spotFiles.append(file)

def search(i,names):
    for s in names:
        if 'rep'+str(i) in s:
            return(s)
    return -1

X_train,y_train=dataPreProcess(indexTrim(X_train),pd.read_csv(rawPath+search(1,spotFiles),delimiter='\t'))
X_val,y_val=dataPreProcess(indexTrim(X_val),pd.read_csv(rawPath+search(2,spotFiles),delimiter='\t'))
X_test,y_test=dataPreProcess(indexTrim(X_test),pd.read_csv(rawPath+search(3,spotFiles),delimiter='\t'))

#Save Processed Data
X_train.to_csv(dataDir+'X_train.csv')
pd.DataFrame(y_train).to_csv(dataDir+'y_train.csv')
X_val.to_csv(dataDir+'X_val.csv')
pd.DataFrame(y_val).to_csv(dataDir+'y_val.csv')
X_test.to_csv(dataDir+'X_test.csv')
pd.DataFrame(y_test).to_csv(dataDir+'y_test.csv')


