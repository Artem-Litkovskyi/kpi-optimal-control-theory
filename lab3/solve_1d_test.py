import numpy as np
import matplotlib.pyplot as plt
from core import ExtremaApproximation


def func(t, x, dxdt):
    return dxdt**2 / 2 - x**2 / 2


def func_dfdx(t, x, dxdt):
    return -x


def func_dfdv(t, x, dxdt):
    return dxdt


def analytic_solution(t):
    return np.sin(t)


def main():
    a = 0
    b = np.pi / 2
    x_a = np.array([0], dtype=np.float64)
    x_b = np.array([1], dtype=np.float64)
    n = 10

    approximation = ExtremaApproximation(func, a, b, x_a, x_b, n, dfdx=func_dfdx, dfdv=func_dfdv)
    t = approximation.get_t()
    approx_x = approximation.optimize(maximize=False, use_gradient=True)
    analytic_x = analytic_solution(t)
    print(t)
    print(approx_x)
    print(analytic_x)

    plt.plot(t, approx_x, label='Approximation', marker='o')
    plt.plot(t, analytic_x, label='Analytic Solution', linestyle='--')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
