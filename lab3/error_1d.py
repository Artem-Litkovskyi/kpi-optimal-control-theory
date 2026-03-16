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


def plot_iterations_vs_error():
    a = 0
    b = np.pi / 2
    x_a = np.array([0], dtype=np.float64)
    x_b = np.array([1], dtype=np.float64)
    n = 10

    approximation = ExtremaApproximation(func, a, b, x_a, x_b, n, dfdx=func_dfdx, dfdv=func_dfdv)
    t = approximation.get_t()
    analytic_x = analytic_solution(t)

    iteration_errors = []

    def callback(xk):
        """Reconstructs the matrix from the flattened parameters and calculates RMSE for x1."""
        current_x = np.hstack((x_a, xk, x_b))
        error_x1 = np.sqrt(np.mean((current_x - analytic_x) ** 2))
        print(error_x1)
        iteration_errors.append(error_x1)

    approximation.optimize(maximize=False, use_gradient=False, zero_x0=True, callback=callback)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(iteration_errors) + 1), iteration_errors, marker='o', color='tab:blue')
    plt.title(f'Iterations vs Error (n={n})')
    plt.xlabel('Iteration Number')
    plt.ylabel('RMSE of x1')
    plt.grid(True, which="both", ls="--")
    plt.show()


def plot_n_vs_error():
    a = 0
    b = np.pi / 2
    x_a = np.array([0], dtype=np.float64)
    x_b = np.array([1], dtype=np.float64)

    # Test various grid resolutions
    n_values = [10, 20, 30, 50, 80, 130, 210]
    final_errors = []

    for n in n_values:
        approximation = ExtremaApproximation(func, a, b, x_a, x_b, n, dfdx=func_dfdx, dfdv=func_dfdv)
        t = approximation.get_t()
        analytic_x = analytic_solution(t)
        approx_x = approximation.optimize(maximize=False, use_gradient=True, zero_x0=False).T.flatten()

        # Calculate RMSE for x1
        error_x1 = np.sqrt(np.mean((approx_x - analytic_x) ** 2))
        final_errors.append(error_x1)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, final_errors, marker='s', color='tab:red')
    # plt.yscale('log')
    plt.title('Number of Points (n) vs Final Error')
    plt.xlabel('Number of Discrete Points (n)')
    plt.ylabel('Final RMSE of x1')
    plt.grid(True, which="both", ls="--")
    plt.show()


if __name__ == "__main__":
    print("Running Iteration test...")
    plot_iterations_vs_error()

    print("Running Resolution (n) test...")
    plot_n_vs_error()