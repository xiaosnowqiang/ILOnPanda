import numpy as np
import matplotlib.pyplot as plt


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


class QuatDMP_Agent():
    def __init__(self, n_bfs, timesteps, dt=.001, y0=np.array([1, 0, 0, 0]), goal=np.array([1, 0, 0, 0]),
                 w=None, ay=None, by=None, **kwargs):
        self.n_bfs = n_bfs
        self.dt = dt

        self.y0 = np.array(y0)
        self.goal = np.array(goal)

        if w is None:
            # default is f = 0
            w = np.zeros((3, self.n_bfs))
        self.w = w
        self.ay = 48. if ay is None else ay  # Schaal 2012
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
        for d in range(4):
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
        self.dy = np.zeros(3)
        self.ddy = np.zeros(3)
        self.cs.reset_state()

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on the forcing term.

                x float: the current value of the canonical system
                dmp_num int: the index of the current dmp
                """
        return x * quat_err(self.goal,self.y0)[dmp_num]

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
        self.w = np.zeros((3, self.n_bfs))
        for d in range(3):
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
        """

        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        self.goal = self.gen_goal(y_des)

        self.check_offset()

        # calculate angle velocity of y_des
        dy_des = cal_angle_vel(y_des,self.dt)
        dy_des = np.hstack((dy_des,np.zeros((3, 1))))
        # calculate acceleration of y_des
        ddy_des = np.diff(dy_des) / self.dt
        # add zero to the beginning of every row
        ddy_des = np.hstack((ddy_des,np.zeros((3, 1))))

        f_target = np.zeros((y_des.shape[1], 3))
        # find the force required to move along this trajectory
        for i in range(self.timesteps):
            f_target[i, :] = (ddy_des[:,i] - self.ay *
                              (self.by * quat_err(self.goal,y_des[:,i]) -
                               dy_des[:,i]))

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
        y_track = np.zeros((timesteps, 4))
        dy_track = np.zeros((timesteps, 3))
        ddy_track = np.zeros((timesteps, 3))
        for t in range(timesteps):
            # run and record timestep
            y_track[t,:], dy_track[t,:], ddy_track[t,:] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def step(self, tau=1.0):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        """

        # run canonical system
        x = self.cs.step(tau=tau)

        # generate basis function activation
        psi = self.gen_psi(x)

        for d in range(3):
            # generate the forcing term
            f = (self.gen_front_term(x, d) *
                 (np.dot(psi, self.w[d,:])) / np.sum(psi))
            # DMP acceleration
            self.ddy[d] = (self.ay*(self.by* quat_err(self.goal,self.y)[d]-self.dy[d] / tau) + f) * tau

            self.dy[d] += self.ddy[d] * tau * self.dt

        temp = np.zeros((4,))
        temp[1:] = self.dy
        temp_quat_exp = quat_exp(temp * self.dt/(2*tau))

        v,u = quat_mul(temp_quat_exp,self.y)
        self.y[0] = v
        self.y[1:] = u
        self.y = self.y / np.linalg.norm(self.y)

        return self.y, self.dy, self.ddy


def quat_mul(q,p):
    v = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
    x = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2]
    y = q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1]
    z = q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
    u = np.array([x,y,z])
    return v,u

def quat_err(q1,q2):
    q2[1:] = -q2[1:]
    v,u = quat_mul(q1,q2)
    err = 2*np.arccos(v)*u / (np.linalg.norm(u)+1e-6)
    return err

def quat_exp(q):
    norm = np.linalg.norm(q)
    quat_exp = np.cos(norm) + np.sin(norm)/norm * q
    return quat_exp

def cal_angl_vel_base(q):
    return np.array([[-q[1],q[0],-q[3],q[2]],
                     [-q[2],q[3],q[0],-q[1]],
                     [-q[3],-q[2],q[1],q[0]]])

def cal_angle_vel(poses,dt):
    length = np.shape(poses)[1]
    ang_vel = []
    for i in range(1,length):
        ang_vel.append(2*cal_angl_vel_base(poses[:,i-1]) @ (poses[:,i] - poses[:,i-1])/dt)

    ang_vel = np.array(ang_vel).T
    return ang_vel


