#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8]{inputenc}",    
         r"\usepackage[russian]{babel}",  
    ])
})

data = pd.read_csv(sys.argv[1], header=None)

fig = plt.figure(figsize=(5,4))

plt.plot(data.values[:,0],data.values[:,1])

plt.xlabel("Задержка гироскопа (мс)")
plt.ylabel("Функция потерь")

fig.tight_layout()

plt.show()

plt.savefig('fig.pgf', backend='pgf')