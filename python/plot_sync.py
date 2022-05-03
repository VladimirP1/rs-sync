#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

plt.rc('axes', labelsize=14)

fps = 60
ax2_yrange = 10

data2 = pd.read_csv(sys.argv[1], header=None)
fig, ax1 = plt.subplots(figsize=(8,4.5))
ax2 = ax1.twinx()
ax1.set_ylim(0, 2)

r = st.linregress(data2.values[:, 0], data2.values[:, 1])
ndata = r.intercept + r.slope * data2.values[:, 0]

ax2_ymid = (np.min(ndata) + np.max(ndata)) / 2
ax2.set_ylim(ax2_ymid - ax2_yrange, ax2_ymid + ax2_yrange)

leg1, leg2 = [], []

leg1.append(ax1.plot(data2.values[:, 0]/fps, np.abs(ndata -
            data2.values[:, 1]), color='green', alpha=.3, label='difference')[0])
leg2.append(ax2.plot(data2.values[:, 0]/fps, ndata, color='orange',
            alpha=.7, label='least squares fit on gyro delay')[0])
leg2.append(ax2.plot(data2.values[:, 0]/fps, data2.values[:,
            1], color='red', label='gyro delay')[0])

ax1.set_xlabel("Video time (sec)")
ax2.set_ylabel("Gyro delay(ms)")
ax1.set_ylabel("Gyro delay error (ms)")

ax2.grid(axis="y")
ax1.grid(axis="x")

ax1.legend(handles=leg1, loc="upper left")
ax2.legend(handles=leg2, loc="upper right")

rmse = np.std(ndata-data2.values[:, 1])
plt.text(.8, -.15, "RMSE={:.3f}".format(rmse),
         color="darkred", size=14, transform=ax1.transAxes)

fig.tight_layout()

print('rms error:', rmse)

plt.show()