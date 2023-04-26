from SPINAI.utils import *
import sys
import os

args=sys.argv
exp=sys.argv[1]
pathToAttributions=exp+'/Attribution/'
meanImpThresh=args[2]
PNIThresh=args[3]

#Concatenate Attributions
attributions=[]
for file in os.listdir(pathToAttributions):
  attributions.append(pd.read_csv(pathToAttributions+file,index_col=0))
attMatrix=abs(pd.concat(attributions,axis=0))

#Get Mean Importance
meanImp=attMatrix.mean(axis=0)

#Get Mean Non-Zero Importance
geneSums=attMatrix.sum(axis=0)
zeroMask=attMatrix>0
nonZeroSum=zeroMask.sum(axis=0)
MNI=geneSums/nonZeroSum

#Get Proportion of Non-Zero Importance
PNI=nonZeroSum/attMatrix.shape[0]

#Get SPGs
spgTable=pd.DataFrame((meanImp,MNI,PNI)).T
spgTable.columns=['meanImportance','MNI','PNI']

isSPG=spgTable.meanImportance>meanImpThresh
minMNI=spgTable.loc[isSPG].MNI.min()
isSPG=isSPG | ((MNI>minMNI) & (PNI>PNIThresh))
spgTable['isSPG']=isSPG

#Write
spgTable.to_csv(pathToAttributions+'spgTable.csv')

