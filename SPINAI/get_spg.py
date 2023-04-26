from SPINAI.utils import *
import sys
import os

args=sys.argv
pathToAttributions=args[1]

attributions=[]
for file in os.listdir(pathToAttributions):
  attributions.append(pd.read_csv(file))

attMatrix=np.vstack(attributions)
