import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

plt.rc('axes',  titlesize=14)
plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)   
plt.rc('ytick', labelsize=10)    

ax2_yrange = 10

fpss = [30, 30, 60, 60]

titles = [
    "trail running 1",
    "Упрощённая модификация, trail running 1",
    "fpv flight 1",
    "Упрощённая модификация, fpv flight 1"
]

files = ["sync_GH011230.csv", "_sync_GH011230.csv",
         "sync_GX012440.csv", "_sync_GX012440.csv"]

fig, axz = plt.subplots(2, 2, figsize=(14, 9))
print(type(axz))
for ax1, file, title, fps in zip(axz.flatten(), files, titles, fpss):
    ax1.set_title(title)
    data2 = pd.read_csv(file, header=None)
    ax2 = ax1.twinx()
    ax1.set_ylim(0, 1)

    r = st.linregress(data2.values[:, 0], data2.values[:, 1])
    ndata = r.intercept + r.slope * data2.values[:, 0]

    ax2_ymid = (np.min(ndata) + np.max(ndata)) / 2
    ax2.set_ylim(ax2_ymid - ax2_yrange, ax2_ymid + ax2_yrange)

    line_difference, = ax1.plot(data2.values[:, 0]/fps, np.abs(ndata -
                                                               data2.values[:, 1]), color='green', alpha=.5, label='разность')
    line_lsq_fit, = ax2.plot(data2.values[:, 0]/fps, ndata, color='orange',
                             alpha=.5, label='линейная регрессия')

    line_delay, = ax2.plot(data2.values[:, 0]/fps, data2.values[:,
                                                                1], color='red', alpha=.5, label='задержка гироскопа')
    ax1.grid(axis='x')
    ax2.grid(axis='y')

    ax1.set_xlabel("Позиция видео (сек)")
    ax2.set_ylabel("Задержка гироскопа (мс)")
    ax1.set_ylabel("Ошибка задержки гироскопа (мс)")

    # ax1.legend(handles=leg1, loc="upper left")
    # ax2.legend(handles=leg2, loc="upper right")

    rmse = np.std(ndata-data2.values[:, 1])
    plt.text(.8, -.15, "RMSE={:.3f}".format(rmse),
             color="darkred", size=14, transform=ax1.transAxes)
    print('rms error:', rmse)


fig.tight_layout()
fig.subplots_adjust(bottom=0.1)
fig.legend(handles=[line_delay], loc=(0, 0))
fig.legend(handles=[line_lsq_fit], loc=(.4, 0))
fig.legend(handles=[line_difference], loc=(.8, 0))


plt.show()

plt.savefig('fig.png')
