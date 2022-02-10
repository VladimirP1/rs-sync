
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


csv = pd.read_csv("out.csv")
print(csv.values.shape)
plt.figure()
# plt.xlim(-.1,.1)
plt.plot(csv.values[:,0], csv.values[:,1])
plt.savefig("a.png")

csv = pd.read_csv("fine_cost.csv")
print(csv.values.shape)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# plt.xlim(-.1,.1)
ax1.plot(csv.values[:,0], csv.values[:,1])
ax2.plot(csv.values[:,0], csv.values[:,2])
plt.savefig("cost.png")

csv = pd.read_csv("fine_cost0.csv")
print(csv.values.shape)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# plt.xlim(-.1,.1)
ax1.plot(csv.values[:,0], csv.values[:,2])
ax2.plot(csv.values[:,0], csv.values[:,3])
plt.savefig("cost0.png")

csv = pd.read_csv("trace.csv")
print(csv.values.shape)
plt.figure()
plt.plot(csv.values[:,0],csv.values[:,1])
plt.plot(csv.values[:,0],csv.values[:,4])
plt.savefig("b0.png")
plt.figure()
plt.plot(csv.values[:,0],csv.values[:,2])
plt.plot(csv.values[:,0],csv.values[:,5])
plt.savefig("b1.png")
plt.figure()
plt.plot(csv.values[:,0],csv.values[:,3])
plt.plot(csv.values[:,0],csv.values[:,6])
plt.savefig("b2.png")

