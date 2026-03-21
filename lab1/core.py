import numpy as np
from scipy.integrate import solve_ivp


class BoatOptimization:
    def __init__(self, target_x, target_y, boat_v, flow_x, smart=False):
        """
        :param target_x: Target X coordinate
        :param target_y: Target Y coordinate
        :param boat_v: Boat speed relative to water
        :param flow_x: Flow velocity along X axis, can be a constant or a function of y
        :param smart: If True, use optimal control strategy; if False, head directly towards the target
        """

        self.target_x = target_x
        self.target_y = target_y
        self.boat_v = boat_v
        self.smart = smart

        if callable(flow_x):
            self.flow_x_func = flow_x
        else:
            self.flow_x_func = lambda y: flow_x

    def reachable_points_bound(self, x_span, max_step=0.1):
        """
        Solves y' = boat_v / sqrt(flow_x_func(y)^2 - boat_v^2)

        :param x_span: Tuple (x_start, x_end) for integration
        :param max_step: scipy.integrate.solve_ivp max_step parameter to control integration accuracy
        :return: Scipy ODE solution object
        """

        def dydx(x, y):
            flow = self.flow_x_func(y[0])
            
            if flow <= self.boat_v:
                raise ValueError(f'Flow velocity must be greater than boat speed for a valid solution.')

            return self.boat_v / np.sqrt(flow ** 2 - self.boat_v ** 2)

        sol = solve_ivp(dydx, x_span, [0], dense_output=True, max_step=max_step)
        return sol

    def trajectory(self, allowed_dist, t_span=(0, 1000), max_step=0.1):
        """
        Solves the kinematic system until the boat is within allowed_dist of the target.

        :param allowed_dist: Distance threshold to stop integration
        :param t_span: Tuple (t_start, t_end) max time span for integration
        :param max_step: scipy.integrate.solve_ivp max_step parameter to control integration accuracy
        :return: Scipy ODE solution object
        """

        def system(t, state):
            x, y = state

            dist = np.sqrt((self.target_x - x) ** 2 + (self.target_y - y) ** 2)

            if dist == 0:
                return [0.0, 0.0]

            flow = self.flow_x_func(y)

            if self.smart:
                alpha = np.arctan2(self.target_y - y, self.target_x - x)

                half_sqrt_d = np.sqrt(self.boat_v ** 2 - (flow * np.sin(alpha)) ** 2)
                net_velocity = flow * np.cos(alpha) + half_sqrt_d

                uy = net_velocity / self.boat_v * np.sin(alpha)
                ux = - (net_velocity ** 2 - flow ** 2 - self.boat_v ** 2) / (2 * net_velocity * flow)
            else:
                ux = (self.target_x - x) / dist
                uy = (self.target_y - y) / dist

            dxdt = flow + self.boat_v * ux
            dydt = self.boat_v * uy

            return [dxdt, dydt]

        def target_reached(t, state):
            x, y = state
            dist = np.sqrt((self.target_x - x) ** 2 + (self.target_y - y) ** 2)
            return dist - allowed_dist

        target_reached.terminal = True  # Tell solve_ivp to terminate when target_reached returns 0
        target_reached.direction = -1  # Trigger only when distance is decreasing

        sol = solve_ivp(system, t_span, [0, 0], events=target_reached, dense_output=True, max_step=max_step)

        return sol