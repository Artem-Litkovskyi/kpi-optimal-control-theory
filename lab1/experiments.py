from core import BoatOptimization
from output import create_plot


N = 13


def main():
    cases = (  # N * np.cos(N * np.pi / 25)
        (BoatOptimization(5, 3, 5, lambda y: 6+y), 'v=5, s(y)=6+y', 'test'),
    )

    for case in cases:
        opt, title, filename = case
        
        try:
            reach_sol = opt.reachable_points_bound((0, opt.target_x * 2))
        except ValueError as e:
            print(f'Error for {filename}: {e}')
            reach_sol = None
        
        traj_sol = opt.trajectory(0.1, t_span=(0, 1000))

        create_plot(reach_sol, traj_sol, opt.target_x, opt.target_y, title, filename)


if __name__ == '__main__':
    main()