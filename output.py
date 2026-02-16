import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def create_plot_two_models(res1, res2, label1, label2, path, filename, population1=None, population2=None):
    plt.figure(figsize=(7, 4))
    plt.xlabel('Дні')
    plt.ylabel('Кількість індивідів')
    plt.grid()

    end_idx = max(res1['end_idx'], res2['end_idx'])
    plt.xlim(0, res1['t'][end_idx])

    linestyles = ('--', '-')

    curves = {
        'S': 'tab:blue',
        'I': 'tab:red',
        'R': 'tab:green',
        'D': 'black',
    }

    if 'Q' in res1 or 'Q' in res2:
        curves['Q'] = 'tab:orange'

    for res, n, lbl, ls in zip((res1, res2), (population1, population2), (label1, label2), linestyles):
        for c in curves.keys():
            if c not in res:
                continue

            data = res[c].copy()
            if n is not None:
                data /= n

            plt.plot(res['t'], res[c], label=f'{lbl}: {c}', color=curves[c], linestyle=ls)

    custom_legend = [
        Line2D([], [], color='gray', linestyle=linestyles[0], label=label1),
        Line2D([], [], color='gray', linestyle=linestyles[1], label=label2),
    ] + [Patch(color=color, label=key) for key, color in curves.items()]

    plt.legend(handles=custom_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + filename + '.png')


def create_table_two_models(res1, res2, label1, label2, path, filename):
    table_data = [
        ['Параметр'],
        ['Тривалість епідемії (дні)'],
        ['Пік інфікованих'],
        ['День піку інфікованих'],
        ['S на кінець епідемії'],
        ['I на кінець епідемії'],
        ['R на кінець епідемії'],
        ['D на кінець епідемії'],
    ]

    if 'Q' in res1 or 'Q' in res2:
        table_data.append(['Q на кінець епідемії'])

    for res, lbl in zip((res1, res2), (label1, label2)):
        end_idx = res['end_idx']
        peak_infected_idx = res['peak_infected_idx']
        peak_infected = res['peak_infected']
        table_data[0].append(lbl)
        table_data[1].append(res['t'][end_idx])
        table_data[2].append(peak_infected)
        table_data[3].append(res['t'][peak_infected_idx])
        table_data[4].append(res['S'][end_idx])
        table_data[5].append(res['I'][end_idx])
        table_data[6].append(res['R'][end_idx])
        table_data[7].append(res['D'][end_idx])
        if len(table_data) > 8:
            table_data[8].append(res['Q'][end_idx] if 'Q' in res else 0)

    if not os.path.exists(path):
        os.makedirs(path)

    np.savetxt(path + filename + '.csv', table_data, delimiter=',', fmt='%s')