
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


csv = pd.read_csv("log.csv")
print(csv.values.shape)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# plt.xlim(-.1,.1)
# ax1.set_ylim(0,5e-10)
ax1.plot(csv.values[:,0], csv.values[:,1])
#ax2.plot(csv.values[:,0], csv.values[:,2], color='orange')
# plt.plot(csv.values[:,0], csv.values[:,3])
plt.savefig("a.png")

