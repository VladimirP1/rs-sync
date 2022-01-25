
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


csv = pd.read_csv("out.csv")
print(csv.values.shape)
plt.figure()
plt.plot(csv.values[:,0], csv.values[:,1])
plt.savefig("a.png")

csv = pd.read_csv("trace.csv")
print(csv.values.shape)
plt.figure()
plt.plot(csv.values[:,0])
plt.plot(csv.values[:,3])
plt.savefig("b0.png")
plt.figure()
plt.plot(csv.values[:,1])
plt.plot(csv.values[:,4])
plt.savefig("b1.png")
plt.figure()
plt.plot(csv.values[:,2])
plt.plot(csv.values[:,5])
plt.savefig("b2.png")

