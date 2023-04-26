from SPINAI.utils import *
import numpy as np
import pandas as pd
import sys
import os

outDir=sys.argv[1]
subDirs=os.listdir(outDir)

for subDir in subDirs:
    s=list(subDir)
    s[5]=' '
    s="".join(s)
    paramCombos=pd.read_csv(outDir+'/'+subDir+'/'+'cv_results.csv')
    optimal=np.argmin(paramCombos['CV_Error'])
    best_l=paramCombos['N Layers'][optimal]
    best_lr=paramCombos['Learning Rate'][optimal]
    bestPred=pd.read_csv(outDir+'/'+subDir+'/'+str(best_l)+'_'+str(best_lr)+'/predictions.csv',index_col=0).to_numpy()
    makeErrorPlotsFromData(bestPred[:,0:2], bestPred[:,2:4], None, s, (0,40),plotPredicted=False,savePath=outDir+'/'+subDir+'/'+'pred.png')


