import numpy as np

class Canonical_System():
    def __init__(self, dt, timesteps, ax=1.0):
        self.ax = ax
        self.step = self.step_discrete
        self.run_time = dt * timesteps
        self.dt = dt
        self.timesteps = timesteps
        self.reset_state()

    def rollout(self, **kwargs):
        if 'tau' in kwargs:
            timesteps = int(self.timesteps / kwargs['tau'])
        else:
            timesteps = self.timesteps
        self.x_track = np.zeros(timesteps)
        self.reset_state()

        for t in range(timesteps):
            self.x_track[t] = self.x
            self.step(**kwargs)
        return self.x_track

    def reset_state(self):
        self.x = 1.0

    def step_discrete(self, tau=1.0, error_coupling=1.0):
        self.x += (-self.ax * self.x * error_coupling) * tau * self.dt
        return self.x


class DMP_Agent():
    def __init__(self, n_dmps, n_bfs, timesteps, dt=.001, y0=[0, 0, 0, 0, 0, 0], goal=[1, 1, 1, 1, 1, 1],
                 w=None, ay=None, by=None, **kwargs):
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt

        self.y0 = np.array(y0)
        self.goal = np.array(goal)

        if w is None:
            # default is f = 0
            w = np.zeros((self.n_dmps, self.n_bfs))
        self.w = w
        self.ay = np.ones(n_dmps) * 25. if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4. if by is None else by  # Schaal 2012

        # set up the CS
        self.cs = Canonical_System(dt=self.dt, timesteps=timesteps, **kwargs)
        self.timesteps = timesteps

        # set up the DMP system
        self.reset_state()

        # set mean of Gaunssian basis functions -> self.c
        self.gen_centers()
        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.cs.ax
        self.check_offset()

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""
        for d in range(self.n_dmps):
            if (self.y0[d] == self.goal[d]):
                self.goal[d] += 1e-4

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        '''
        centers for first-order canonical system
        so here we used exponential function directly
        '''

        '''
        x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]'''

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on the forcing term.

                x float: the current value of the canonical system
                dmp_num int: the index of the current dmp
                """
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        """Generate the goal for path imitation."""

        return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
                canonical system rollout.

                x float, array: the canonical system state or path
                """
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c) ** 2)

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
                that the target forcing term trajectory is matched.

                f_target np.array: the desired forcing term trajectory
                """

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track ** 2 * psi_track[:, b])
                self.w[d, b] = numer / (k * denom)
        self.w = np.nan_to_num(self.w)

    def imitate_path(self, y_des):
        """Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.

        y_des list/array: the desired trajectories of each DMP
                          should be shaped [n_dmps, run_time]
        """

        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        self.goal = self.gen_goal(y_des)

        self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate
        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(self.timesteps):
                path[d, t] = path_gen(t * self.dt)
        y_des = path

        # calculate velocity of y_des
        dy_des = np.diff(y_des) / self.dt
        # add zero to the beginning of every row
        dy_des = np.hstack((np.zeros((self.n_dmps, 1)), dy_des))

        # calculate acceleration of y_des
        ddy_des = np.diff(dy_des) / self.dt
        # add zero to the beginning of every row
        ddy_des = np.hstack((np.zeros((self.n_dmps, 1)), ddy_des))

        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        # find the force required to move along this trajectory
        for d in range(self.n_dmps):
            f_target[:, d] = (ddy_des[d] - self.ay[d] *
                              (self.by[d] * (self.goal[d] - y_des[d]) -
                               dy_des[d]))

        # efficiently generate weights to realize f_target
        self.gen_weights(f_target)

        self.reset_state()
        return self.w

    def rollout(self, timesteps=None, goal=None, y0=None, **kwargs):
        """Generate a system trial, no feedback is incorporated."""

        if goal is not None:
            self.goal = goal

        if y0 is not None:
            self.y0 = y0

        self.reset_state()

        if timesteps is None:
            if 'tau' in kwargs:
                timesteps = int(self.timesteps / kwargs['tau'])
            else:
                timesteps = self.timesteps

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))
        for t in range(timesteps):
            # run and record timestep
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def step(self, tau=1.0, error=0.0, external_force=None):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        """

        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # generate basis function activation
        psi = self.gen_psi(x)

        for d in range(self.n_dmps):

            # generate the forcing term
            f = (self.gen_front_term(x, d) *
                 (np.dot(psi, self.w[d])) / np.sum(psi))
            # DMP acceleration
            self.ddy[d] = (self.ay[d] *
                           (self.by[d] * (self.goal[d] - self.y[d]) -
                            self.dy[d] / tau) + f) * tau
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * self.dt * error_coupling

        return self.y, self.dy, self.ddy

    def gen_force_term(self, timesteps=None, goal=None, y0=None, **kwargs):
        if goal is not None:
            self.goal = goal

        if y0 is not None:
            self.y0 = y0

        self.reset_state()

        if timesteps is None:
            if 'tau' in kwargs:
                timesteps = int(self.timesteps / kwargs['tau'])
            else:
                timesteps = self.timesteps

        f = np.zeros((self.timesteps, self.n_dmps))
        tau = 1.0
        error_coupling = 1.0
        for i in range(self.timesteps):
            # run canonical system
            x = self.cs.step(tau=tau, error_coupling=error_coupling)

            # generate basis function activation
            psi = self.gen_psi(x)

            for d in range(self.n_dmps):
                # generate the forcing term
                f[i, d] = (self.gen_front_term(x, d) *
                           (np.dot(psi, self.w[d])) / np.sum(psi))
        return f