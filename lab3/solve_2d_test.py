import numpy as np
import matplotlib.pyplot as plt
from core import ExtremaApproximation


def func(t, x, dxdt):
    x1, x2 = x
    dx1dt, dx2dt = dxdt
    return (dx1dt ** 2 + dx2dt ** 2) / 2 - 9.81 * x2


def func_dfdx(t, x, dxdt):
    return np.array([0, -9.81], dtype=np.float64)


def func_dfdv(t, x, dxdt):
    dx1dt, dx2dt = dxdt
    return np.array([dx1dt, dx2dt], dtype=np.float64)


def analytic_solution(t):
    x1 = t
    x2 = 9.81 / 2 * t * (10 - t)
    return np.array([x1, x2], dtype=np.float64)


def main():
    a = 0
    b = 10
    x_a = np.array([0, 0], dtype=np.float64)
    x_b = np.array([10, 0], dtype=np.float64)
    n = 50

    approximation = ExtremaApproximation(func, a, b, x_a, x_b, n, dfdx=func_dfdx, dfdv=func_dfdv)
    t = approximation.get_t()
    approx_x = approximation.optimize(maximize=False, use_gradient=True)
    analytic_x = analytic_solution(t).T
    print(t)
    print(approx_x)
    print(analytic_x)

    plt.plot(t, approx_x.T[0], label='Approximation x1', marker='o', color='tab:red')
    plt.plot(t, analytic_x.T[0], label='Analytic Solution x1', linestyle='--', color='tab:red')
    plt.plot(t, approx_x.T[1], label='Approximation x2', marker='o', color='tab:green')
    plt.plot(t, analytic_x.T[1], label='Analytic Solution x2', linestyle='--', color='tab:green')
    plt.xlabel('t')
    plt.ylabel('xi(t)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
