import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1], header=None)

#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#ax1.plot(data.values[:,0],data.values[:,1])
#ax1.plot(data.values[:,3],data.values[:,4],'o')
#ax2.plot(data.values[:,0],data.values[:,2], color='red', alpha = .5)

plt.plot(data.values[:,0],data.values[:,1])
#ift = 150/2
#plt.plot(data.values[:,0]-ift,data.values[:,1],color='orange')
#plt.plot(data.values[:,0]+ift,data.values[:,1],color='orange')
#plt.plot(data.values[:,0],data.values[:,2], alpha=.4)
#plt.plot(data.values[:,4],data.values[:,3], 'o')
plt.show()
