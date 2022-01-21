
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


csv = pd.read_csv("data.csv")

print(csv.values.shape)

# plt.plot(csv.values[:,0])
# plt.plot(csv.values[:,3])
# plt.plot(csv.values[:,2]/7000)
# plt.savefig("a.png")

plt.plot(csv.values[:,0], csv.values[:,1])
# plt.xlim(.02,.06)
plt.savefig("a.png")