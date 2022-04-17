import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[russian]{babel}",
    ])
})

plt.rc('axes',  titlesize=14)
plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

ax2_yrange = 10

fpss = [60, 60, 60, 60]

titles = [
    "table, with translation",
    "table, no translation",
    "fpv flight 1, with translation",
    "fpv flight 1, no translation"
]

files = ["sync_GX012439.csv", "_sync_GX012439.csv",
         "sync_GX012440.csv", "_sync_GX012440.csv"]

fig, axz = plt.subplots(2, 2, figsize=(9, 9))
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
                                                               data2.values[:, 1]), color='green', alpha=.5, label='difference')
    line_lsq_fit, = ax2.plot(data2.values[:, 0]/fps, ndata, color='orange',
                             alpha=.5, label='least squares fit on gyro delay')

    line_delay, = ax2.plot(data2.values[:, 0]/fps, data2.values[:,
                                                                1], color='red', alpha=.5, label='gyro delay')

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
fig.legend(handles=[line_lsq_fit], loc=(.2, 0))
fig.legend(handles=[line_difference], loc=(.6, 0))


plt.show()

plt.savefig('fig.pgf', backend='pgf')
