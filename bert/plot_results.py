import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
search_dir = 'results/100-various/'

directory = Path(search_dir)
files = [str(file) for file in directory.glob('*') if str(file).endswith('.npy')]
files = sorted(files)

import re

for file in files:
    results = np.load(file)

    data = results[1:].T.astype(float).tolist()

    columns = results[0, :]
    columns = ['Summ Screen FD', 'QMSUM', 'QASPER Title']

    highlight_indexes = [len(data[0])-1] * np.size(data, 0)

    fig, ax = plt.subplots()

    ax.boxplot(data, widths=0.5)

    ax.plot(1, data[0][highlight_indexes[0]], 'ro', label='Validation Split')
    for i in range(1,len(data)):
        ax.plot(i+1, data[i][highlight_indexes[i]], 'ro')

    plt.xticks(list(range(1, len(columns)+1)), columns)


    ax.set_ylabel('nDCG@10')
    ax.set_ylim(0, 1)

    name = file.split('/')[-1].replace('.npy', '')
    subtitle = f'nDCG@10 computed for resamples (n={highlight_indexes[0]}) of train split sized to match validation split for SCROLLS datasets on {name}'
    pattern = r'.{1,64}(?:\s+|\b)'
    subtitle = re.sub(pattern, lambda m: m.group(0) + '\n', subtitle).strip()
    plt.suptitle(subtitle)
    ax.legend(loc='lower left')

    plt.tight_layout()

    plt.savefig(f'{search_dir}/{name}.png')

    plt.show()
