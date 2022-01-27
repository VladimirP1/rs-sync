
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


csv = pd.read_csv("out.csv")
print(csv.values.shape)
plt.figure()
plt.plot(csv.values[:,0], csv.values[:,1])
plt.plot(csv.values[:,0], csv.values[:,2])
plt.plot(csv.values[:,0], csv.values[:,3])
#plt.ylim(-.1,.1)
plt.savefig("a.png")

