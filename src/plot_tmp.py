
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# exp = pd.read_csv('GX019642_Hero6 Black-SHUT.csv')
# print(exp.values[:,2])
# left = np.searchsorted(exp.values[:,0], 1100+50)
# right = np.searchsorted(exp.values[:,0], (7140+50)/50*1000)
# print(left,right)

csv = pd.read_csv("sync.csv", header=None)
print(csv.values.shape)

# fig, ax1 = plt.subplots()
# ax2 = ax1.twiny().twinx()
# ax1.set_ylim(-.04,-.025)
# ax2.set_ylim(.009,.004)
# ax1.set_xlim(1100, 7140)
# ax2.set_xlim(exp.values[left,0], exp.values[right-1,0])
#plt.ylim(-.035,-.025)
# ax2.plot(exp)
plt.plot(csv.values[:,0]/30, csv.values[:,1]*1000, color='red')
# ax2.plot(exp.values[left:right,0], exp.values[left:right,2])
# plt.xlabel("sync point position (seconds)")
# plt.ylabel("estimated gyro shift (seconds)")
plt.savefig("a.png")

# csv = pd.read_csv("trace.csv")
# print(csv.values.shape)
# plt.figure()
# plt.plot(csv.values[:,0])
# plt.plot(csv.values[:,3])
# plt.savefig("b0.png")
# plt.figure()
# plt.plot(csv.values[:,1])
# plt.plot(csv.values[:,4])
# plt.savefig("b1.png")
# plt.figure()
# plt.plot(csv.values[:,2])
# plt.plot(csv.values[:,5])
# plt.savefig("b2.png")

