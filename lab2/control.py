import numpy as np

from core import EpidemicModel
from output import create_plot_two_models, create_table_two_models

PLOT_PATH = 'plots/control/'
TABLE_PATH = 'tables/control/'


def vac_simple_func(t, S, I, R, D, Q):
    return I / (I + 500)


def vac_cos_func(t, S, I, R, D, Q):
    vac_start = 5
    vac_period = 5
    if t < vac_start:
        return 0
    return (1 - np.cos(2 * np.pi / vac_period * (t - vac_start))) / 2


def main():
    cases = (
        (EpidemicModel(), EpidemicModel(vac_rate=0.1), 30, 'Без вакцинації', 'Проста функція', 'vac_simple'),
        (EpidemicModel(contact_rate=1), EpidemicModel(contact_rate=1, vac_rate=0.5), 100, 'Без вакцинації', 'Періодична функція', 'vac_cos'),
        (EpidemicModel(), EpidemicModel(use_Q=True), 100, 'Без тестування', 'З тестуванням', 'quarantine'),
    )

    cases[0][1].set_functions(vac_rate=vac_simple_func)
    cases[1][1].set_functions(vac_rate=vac_cos_func)

    for case in cases:
        m1, m2, days, lbl1, lbl2, filename = case

        res1 = m1.solve_w_stats(days=days, infected_threshold=0.001)
        res2 = m2.solve_w_stats(days=days, infected_threshold=0.001)

        create_plot_two_models(res1, res2, lbl1, lbl2, PLOT_PATH, filename, population1=m1.N, population2=m2.N)
        create_table_two_models(res1, res2, lbl1, lbl2, TABLE_PATH, filename)


if __name__ == '__main__':
    main()