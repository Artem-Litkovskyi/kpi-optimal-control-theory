from core import EpidemicModel
from output import create_plot_two_models

PLOT_PATH = 'plots/feedback/'


def contact_func(t, S, I, R, D, Q):
    panic_threshold = 10000
    if I > panic_threshold:
        return 0.25
    return 1


def inf_func(t, S, I, R, D, Q):
    panic_threshold = 10000
    if I > panic_threshold:
        return 0.7
    return 1


def main():
    cases = (
        (EpidemicModel(contact_rate=10), EpidemicModel(contact_rate=10), 'Без зв\'язку', 'Ізоляція', 'contact_rate'),
        (EpidemicModel(contact_rate=10), EpidemicModel(contact_rate=10), 'Без зв\'язку', 'Інд. захисні з.', 'inf_prob'),
    )

    cases[0][1].set_functions(contact_rate=contact_func)
    cases[1][1].set_functions(inf_prob=inf_func)

    for case in cases:
        m1, m2, lbl1, lbl2, filename = case

        res1 = m1.solve_w_stats(days=30, infected_threshold=0.001)
        res2 = m2.solve_w_stats(days=30, infected_threshold=0.001)

        create_plot_two_models(res1, res2, lbl1, lbl2, PLOT_PATH, filename, population1=m1.N, population2=m2.N)


if __name__ == '__main__':
    main()