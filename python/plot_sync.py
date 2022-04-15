import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


data2 = pd.read_csv(sys.argv[1], header=None)
fps=60
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
#ax1.plot(data.values[:,0],data.values[:,1])
#ax1.plot(data.values[:,3],data.values[:,4],'o')
ax2.plot(data2.values[:,0]/fps,data2.values[:,1], color='red', alpha = .5, label='gyro delay')
ax1.set_ylim(0,1)

r = st.linregress(data2.values[:,0],data2.values[:,1])
ndata = r.intercept + r.slope * data2.values[:,0]
ax1.plot(data2.values[:,0]/fps,np.abs(ndata-data2.values[:,1]), color='green', alpha = .5)
ax2.plot(data2.values[:,0]/fps,ndata, color='orange', alpha = .5, label='least squares fit on gyro delay')
ax2.legend()
print(ndata[0])
print('rms error:', np.std(ndata-data2.values[:,1]))

#plt.plot(data.values[:,0],data.values[:,1])
#plt.plot(data2.values[:,0],data2.values[:,1])
#ift = 150/2
#plt.plot(data.values[:,0]-ift,data.values[:,1],color='orange')
#plt.plot(data.values[:,0]+ift,data.values[:,1],color='orange')
#plt.plot(data.values[:,0],data.values[:,2], alpha=.4)
#plt.plot(data.values[:,4],data.values[:,3], 'o')
plt.show()
