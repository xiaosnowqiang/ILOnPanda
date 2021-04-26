import numpy as np
from scipy import interpolate

class DMP:
    def __init__(self, T, dt, a=150, b=25, n_bfs=40):
        self.T = T
        self.dt = dt
        self.y0 = 0.0
        self.g = 1.0
        self.a = a
        self.b = b
        self.n_bfs = n_bfs

        # canonical system
        a = 1.0
        self.cs = CanonicalSystem(a, T, dt)

        # initialize basis functions for LWR
        self.w = np.zeros(n_bfs)
        self.centers = None
        self.widths = None
        self.set_basis_functions()

        # executed trajectory
        self.y = None
        self.yd = None
        self.z = None
        self.zd = None

        # desired path
        self.path = None

        self.reset()

    def reset(self):
        if self.y0 is not None:
            self.y = self.y0  # .copy()
        else:
            self.y0 = 0.0
            self.y = 0.0
        self.yd = 0.0
        self.z = 0.0
        self.zd = 0.0
        self.cs.reset()

    def set_basis_functions(self):
        time = np.linspace(0, self.T, self.n_bfs)
        self.centers = np.zeros(self.n_bfs)
        self.centers = np.exp(-self.cs.a * time)
        self.widths = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.centers / self.cs.a

    def psi(self, theta):
        if isinstance(theta, np.ndarray):
            theta = theta[:, None]
        return np.exp(-self.widths * (theta - self.centers) ** 2)

    def step(self, tau=1.0, k=1.0, start=None, goal=None):
        if goal is None:
            g = self.g
        else:
            g = goal

        if start is None:
            y0 = self.y0
        else:
            y0 = start

        theta = self.cs.step(tau)
        psi = self.psi(theta)

        f = np.dot(self.w, psi) * theta * k * (g - y0) / np.sum(psi)

        self.zd = self.a * (self.b * (g - self.y) - self.z) + f  # transformation system
        self.zd /= tau

        self.z += self.zd * self.dt

        self.yd = self.z / tau
        self.y += self.yd * self.dt
        return self.y, self.yd, self.z, self.zd

    def fit(self, y_demo, tau=1.0):
        self.path = y_demo
        self.y0 = y_demo[0].copy()
        self.g = y_demo[-1].copy()

        y_demo = interpolate_path(self, y_demo)
        yd_demo, ydd_demo = calc_derivatives(y_demo, self.dt)

        f_target = tau**2 * ydd_demo - self.a * (self.b * (self.g - y_demo) - tau * yd_demo)
        f_target /= (self.g - self.y0)

        theta_seq = self.cs.all_steps()
        psi_funs = self.psi(theta_seq)

        # Locally Weighted Regression
        aa = np.multiply(theta_seq.reshape((1, theta_seq.shape[0])), psi_funs.T)
        aa = np.multiply(aa, f_target.reshape((1, theta_seq.shape[0])))
        aa = np.sum(aa, axis=1)

        bb = np.multiply(theta_seq.reshape((1, theta_seq.shape[0])) ** 2, psi_funs.T)
        bb = np.sum(bb, axis=1)
        self.w = aa / bb

        self.reset()

    def run_sequence(self, tau=1.0, k=1.0, start=None, goal=None):
        y = np.zeros(self.cs.N)
        y[0] = self.y0
        for i in range(self.cs.N):
            y[i], _, _, _ = self.step(tau=tau, k=k, start=start, goal=goal)
        return y


class CanonicalSystem:
    def __init__(self, a, T, dt):
        self.a = a
        self.T = T
        self.dt = dt

        self.time = np.arange(0, T, dt)
        self.N = self.time.shape[0]

        self.theta = None
        self.reset()

    def reset(self):
        self.theta = 1.0

    def step(self, tau=1.0):
        self.theta = self.theta - self.a * self.dt * self.theta / tau
        return self.theta

    def all_steps(self, tau=1.0):
        return np.array([self.step(tau) for _ in range(self.N)])

    # def step_(self):
    #     self.theta = np.exp(-self.a * self.dt) * self.theta


def interpolate_path(dmp, path):
    time = np.linspace(0, dmp.cs.T, path.shape[0])
    inter = interpolate.interp1d(time, path)
    y = np.array([inter(i * dmp.dt) for i in range(dmp.cs.N)])
    return y


def calc_derivatives(y, dt):
    # velocity
    yd = np.diff(y) / dt
    yd = np.concatenate(([0], yd))

    # acceleration
    ydd = np.diff(yd) / dt
    ydd = np.concatenate(([0], ydd))

    return yd, ydd