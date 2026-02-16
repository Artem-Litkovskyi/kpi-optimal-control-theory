import numpy as np
from scipy.integrate import odeint


class EpidemicModel:
    def __init__(self, N=10e6, I0=1, R0=0, D0=0, Q0=0,
                 alpha=0.07, beta=0.05, alpha_q=0.09, beta_q=0.03,
                 contact_rate=5.0, inf_prob=0.75, vac_rate=0.0, test_rate=0.1,
                 use_Q=False):
        self.N = N
        self.S0 = N - I0 - R0 - D0 - Q0
        self.I0 = I0
        self.R0 = R0
        self.D0 = D0
        self.Q0 = Q0

        self.alpha = alpha
        self.beta = beta
        self.alpha_q = alpha_q
        self.beta_q = beta_q

        self.contact_rate = contact_rate
        self.inf_prob = inf_prob
        self.vac_rate = vac_rate
        self.test_rate = test_rate

        self.use_Q = use_Q

        self.contact_rate_func = None
        self.inf_prob_func = None
        self.vac_rate_func = None
        self.test_rate_func = None

    def set_functions(self, contact_rate=None, inf_prob=None, vac_rate=None, test_rate=None):
        if contact_rate: self.contact_rate_func = contact_rate
        if inf_prob: self.inf_prob_func = inf_prob
        if vac_rate: self.vac_rate_func = vac_rate
        if test_rate: self.test_rate_func = test_rate

    def _deriv(self, y, t):
        if self.use_Q:
            S, I, R, D, Q = y
        else:
            S, I, R, D = y
            Q = 0

        contact_rate = self.contact_rate
        inf_prob = self.inf_prob
        vac_rate = self.vac_rate
        test_rate = self.test_rate

        if self.contact_rate_func:
            contact_rate *= self.contact_rate_func(t, S, I, R, D, Q)
        if self.inf_prob_func:
            inf_prob *= self.inf_prob_func(t, S, I, R, D, Q)
        if self.vac_rate_func:
            vac_rate *= self.vac_rate_func(t, S, I, R, D, Q)
        if self.test_rate_func:
            test_rate *= self.test_rate_func(t, S, I, R, D, Q)

        inf_rate = -contact_rate * np.log(1 - inf_prob) * I / self.N

        dSdt = -(inf_rate * S) - (vac_rate * S)

        dI_from_Q = (test_rate * I) if self.use_Q else 0
        dIdt = (inf_rate * S) - (self.alpha * I) - (self.beta * I) - dI_from_Q

        dR_from_Q = (self.alpha_q * Q) if self.use_Q else 0
        dRdt = (self.alpha * I) + (vac_rate * S) + dR_from_Q

        dD_from_Q = (self.beta_q * Q) if self.use_Q else 0
        dDdt = (self.beta * I) + dD_from_Q

        if self.use_Q:
            dQdt = (test_rate * I) - (self.alpha_q * Q) - (self.beta_q * Q)
            return dSdt, dIdt, dRdt, dDdt, dQdt
        else:
            return dSdt, dIdt, dRdt, dDdt

    def solve(self, days, steps_per_day=1):
        t = np.linspace(0, days, days * steps_per_day)

        if self.use_Q:
            y0 = [self.S0, self.I0, self.R0, self.D0, self.Q0]
        else:
            y0 = [self.S0, self.I0, self.R0, self.D0]

        ret = odeint(self._deriv, y0, t)

        results = {
            't': t,
            'S': ret[:, 0],
            'I': ret[:, 1],
            'R': ret[:, 2],
            'D': ret[:, 3]
        }

        if self.use_Q:
            results['Q'] = ret[:, 4]

        return results

    def solve_w_stats(self, days, steps_per_day=1, infected_threshold=0.1):
        res = self.solve(days, steps_per_day)

        total_I = res['I'].copy()

        if 'Q' in res:
            total_I += res['Q'].copy()

        end_idx = len(total_I) - 1
        for i in range(end_idx, -1, -1):
            if total_I[i] > infected_threshold * self.N or i == 0:
                end_idx = i
                break

        res['end_idx'] = end_idx
        res['peak_infected_idx'] = np.argmax(total_I)
        res['peak_infected'] = np.max(total_I)

        return res
