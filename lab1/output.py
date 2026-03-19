import os
import numpy as np
import matplotlib.pyplot as plt


PATH = 'plots'


def create_plot(reach_sol, traj_sol, target_x, target_y, title, filename):
    plt.figure(figsize=(8, 5))

    plt.scatter([0], [0], color='tab:green', marker='o', s=100, zorder=5, label='Start')
    plt.scatter(target_x, target_y, color='tab:orange', marker='*', s=200, zorder=5, label='Target')

    if traj_sol and traj_sol.success:
        traj_x = traj_sol.y[0]
        traj_y = traj_sol.y[1]

        plt.plot(traj_x, traj_y, color='tab:blue', linewidth=2.5, zorder=4, label='Boat Trajectory')

    if reach_sol and reach_sol.success:
        reach_x = reach_sol.t
        reach_y = reach_sol.y[0]

        plt.plot(reach_x, reach_y, color='tab:red', linestyle='--', linewidth=2, label='Reachable Points Boundary')

        y_top = max(np.max(reach_y), target_y) + 5
        plt.fill_between(
            reach_x,
            reach_y,
            y_top,
            color='tab:red',
            alpha=0.15,
            label='Unreachable Area',
        )

    plt.title(title)
    plt.xlabel('X Coordinate (Downstream)')
    plt.ylabel('Y Coordinate (Cross-stream)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.axis('equal')

    plt.xlim(0, target_x*1.5)
    plt.ylim(0, target_y*1.5)

    plt.tight_layout()

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(f'{PATH}/{filename}.png')