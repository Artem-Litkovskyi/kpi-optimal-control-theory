from core import EpidemicModel
from output import create_plot_two_models


PLOT_PATH = 'plots/initial_params/'


def main():
    cases = (
        (EpidemicModel(contact_rate=0.001), EpidemicModel(contact_rate=50), 'Контакти 0.001', 'Контакти 50', 'contact_rate'),
        (EpidemicModel(inf_prob=0.5), EpidemicModel(inf_prob=0.9), 'Зараження 0.5', 'Зараження 0.9', 'inf_prob'),
        (EpidemicModel(alpha=0.04), EpidemicModel(alpha=0.1), 'Альфа 0.04', 'Альфа 0.1', 'alpha'),
        (EpidemicModel(beta=0.01), EpidemicModel(beta=0.1), 'Бета 0.01', 'Бета 0.1', 'beta'),
        (EpidemicModel(vac_rate=0), EpidemicModel(vac_rate=0.05), 'Вакцинація 0', 'Вакцинація 0.01', 'vac_rate'),
        (EpidemicModel(N=10e3), EpidemicModel(N=10e6), 'Населення 10k', 'Населення 10M', 'population'),
    )

    for case in cases:
        m1, m2, lbl1, lbl2, filename = case

        res1 = m1.solve_w_stats(days=100, infected_threshold=0.001)
        res2 = m2.solve_w_stats(days=100, infected_threshold=0.001)

        print(res2['R'])
        
        create_plot_two_models(res1, res2, lbl1, lbl2, PLOT_PATH, filename, population1=m1.N, population2=m2.N)


if __name__ == '__main__':
    main()