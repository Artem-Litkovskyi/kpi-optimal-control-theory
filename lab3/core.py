import numpy as np
from scipy.optimize import minimize


__all__ = ['ExtremaApproximation']


class ExtremaApproximation:
    def __init__(self, f, a, b, x_a, x_b, n, dfdx=None, dfdv=None):
        self.f = f
        self.a = a
        self.b = b
        self.x_a = x_a
        self.x_b = x_b
        self.n = n
        self.dfdx = dfdx
        self.dfdv = dfdv

    def get_t(self):
        return np.linspace(self.a, self.b, self.n)

    def get_dt(self):
        return (self.b - self.a) / (self.n - 1)

    def get_first_derivative(self, x):
        dx = np.diff(x, axis=0)
        dx = np.vstack((dx, dx[-1]))  # Pad the last row to maintain shape
        dt = self.get_dt()
        return dx / dt

    def get_integral(self, t, x, v):
        f_vals = np.array([self.f(t_val, x_row, v_row) for t_val, x_row, v_row in zip(t, x, v)])
        return np.trapezoid(f_vals.T, dx=self.get_dt())

    def _target_function(self, inner_x_flat, maximize=False):
        inner_x = inner_x_flat.reshape(self.n - 2, len(self.x_a))
        x = np.vstack((self.x_a, inner_x, self.x_b))
        i = self.get_integral(self.get_t(), x, self.get_first_derivative(x))
        return -i if maximize else i

    def _gradient(self, inner_x_flat, maximize=False):
        inner_x = inner_x_flat.reshape(self.n - 2, len(self.x_a))
        x = np.vstack((self.x_a, inner_x, self.x_b))
        t = self.get_t()
        dt = self.get_dt()
        dxdt = self.get_first_derivative(x)

        # Evaluate the partial derivatives row-by-row just like the integral
        df_dx_vals = np.array([self.dfdx(t_val, x_row, v_row) for t_val, x_row, v_row in zip(t, x, dxdt)])
        df_dv_vals = np.array([self.dfdv(t_val, x_row, v_row) for t_val, x_row, v_row in zip(t, x, dxdt)])

        grad = np.zeros_like(inner_x)

        for j in range(1, self.n - 1):
            idx = j - 1  # Map from x index to inner_x index

            # Trapz weight is 1.0 for all inner points
            grad_x = dt * df_dx_vals[j]

            # Account for the chain rule based on the forward difference and padding
            if self.n == 3:
                # Extreme edge case: only one inner point
                grad_v = -df_dv_vals[1] + 0.5 * df_dv_vals[0] - 0.5 * df_dv_vals[2]
            elif j == 1:
                # Leftmost inner point interacts with trapz weight 0.5 at j=0
                grad_v = -df_dv_vals[1] + 0.5 * df_dv_vals[0]
            elif j == self.n - 2:
                # Rightmost inner point interacts with padded derivative dx[-1]
                # and trapz weight 0.5 at j=n-1
                grad_v = -df_dv_vals[self.n - 2] + df_dv_vals[self.n - 3] - 0.5 * df_dv_vals[self.n - 1]
            else:
                # Standard interior points
                grad_v = -df_dv_vals[j] + df_dv_vals[j - 1]

            grad[idx] = grad_x + grad_v

        grad_flat = grad.flatten()
        return -grad_flat if maximize else grad_flat

    def optimize(self, maximize=False, method='Nelder-Mead', options=None, use_gradient=False, zero_x0=False, callback=None):
        if zero_x0:
            inner_x0_flat = np.zeros((self.n - 2) * len(self.x_a))
        else:
            inner_x0_flat = np.linspace(self.x_a, self.x_b, self.n)[1:-1].flatten()
        target_function = lambda inner_x_flat: self._target_function(inner_x_flat, maximize)

        jac = None
        if use_gradient:
            if self.dfdx is None or self.dfdv is None:
                raise ValueError('Gradient functions dfdx and dfdv must be passed to __init__ to use gradients.')

            jac = lambda inner_x_flat: self._gradient(inner_x_flat, maximize)

            # If use_gradient is True but the default gradient-free method is selected, switch to BFGS
            if method == 'Nelder-Mead':
                method = 'BFGS'

        result = minimize(target_function, inner_x0_flat, method=method, jac=jac, options=options, callback=callback)

        x_2d = result.x.reshape(self.n - 2, len(self.x_a))
        return np.vstack((self.x_a, x_2d, self.x_b))
