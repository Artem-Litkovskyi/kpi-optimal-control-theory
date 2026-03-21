import numpy as np

from core import BoatOptimization
from output import create_plot


N = 13


def main():
    cases = (
        (BoatOptimization(N * np.cos(N * np.pi / 25), N * np.sin(N * np.pi / 25), np.sqrt(N), lambda y: np.sqrt(N) * y, False), 'v=√13, s(y)=√13y', 'base'),
        (BoatOptimization(10, 10, np.sqrt(N), lambda y: np.sqrt(N)+1+y/4, False), 'v=√13, s(y)=√13+1+y/4', 'unreachable'),
        (BoatOptimization(10, 6, np.sqrt(N), lambda y: np.sqrt(N)+1+y/4, False), 'v=√13, s(y)=√13+1+y/4', 'simple'),
        (BoatOptimization(10, 4, np.sqrt(N), lambda y: np.sqrt(N)+4+np.cos(y)*3, False), 'v=√13, s(y)=√13+4+3cos(y)','simple_cos'),
        (BoatOptimization(10, 6, 2*np.sqrt(N), lambda y: np.sqrt(N)+1+y/4, False), 'v=2√13, s(y)=√13+1+y/4', 'simple_fast'),
        (BoatOptimization(10, 6, np.sqrt(N), lambda y: np.sqrt(N)+1, False), 'v=√13, s(y)=√13+1', 'simple_const'),
        (BoatOptimization(10, 10, np.sqrt(N), lambda y: np.sqrt(N)+1+y/4, True), 'v=√13, s(y)=√13+1+y/4', 'smart_unreachable'),
        (BoatOptimization(10, 8, np.sqrt(N), lambda y: np.sqrt(N)+1+y/4, True), 'v=√13, s(y)=√13+1+y/4', 'smart_close'),
        (BoatOptimization(10, 8, np.sqrt(N), lambda y: np.sqrt(N)+1+y/4, False), 'v=√13, s(y)=√13+1+y/4', 'simple_close'),
        (BoatOptimization(100, 25, np.sqrt(N), lambda y: np.sqrt(N)+1+y/4, True), 'v=√13, s(y)=√13+1+y/4', 'smart_far'),
        (BoatOptimization(100, 25, np.sqrt(N), lambda y: np.sqrt(N)+1+y/4, False), 'v=√13, s(y)=√13+1+y/4', 'simple_far'),
    )

    for case in cases:
        opt, title, filename = case

        print('Running case:', filename)
        
        try:
            reach_sol = opt.reachable_points_bound((0, opt.target_x * 2))
        except ValueError as e:
            print(f'Error for {filename}: {e}')
            reach_sol = None
        
        traj_sol = opt.trajectory(0.25, t_span=(0, 100))

        create_plot(reach_sol, traj_sol, opt.target_x, opt.target_y, title, filename)


if __name__ == '__main__':
    main()