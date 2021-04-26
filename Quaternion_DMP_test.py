from agents.quat_dmp import QuatDMP
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

T = 11
n = 11000
dt = T / n
demo = np.loadtxt(os.path.join('PouringWater2', 'FinedTrajectories', 'gmm_reference_quaternion.txt')).T
demo_qs = np.zeros_like(demo)
demo_qs[0,:] = demo[1,:]
demo_qs[1,:] = demo[2,:]
demo_qs[2,:] = demo[3,:]
demo_qs[3,:] = demo[0,:]
# demo_qs = directed_slerp(R.random(3, random_state=1).as_quat().T, np.array([0,1,2]), np.linspace(0,2,n))
demo_len = demo_qs.shape[1]
key_times = np.arange(demo_len)

times = np.linspace(0, T, num=n)

dmp = QuatDMP(T, dt, n_bfs=1000, K=1000 * np.eye(3), D=40*np.eye(3))
dmp.fit(demo_qs, tau=1.0)

q = np.zeros((4, n))
qd = np.zeros((4, n))
omega = np.zeros((3, n))
omegad = np.zeros((3, n))
goal1 = np.array([0.59267273,0.59229562,0.28305234 ,0.46669724])
goal2 = np.array([0.24751033,0.71336469,0.50727476,0.41535741])

pour_wine_goal1 = np.array([0.65949504, 0.57595389, 0.32531583, 0.35708965])
pour_wine_goal2 = np.array([0.73132475, 0.47027616, 0.27835707, 0.40807081])
pour_wine_goal3 = np.array([0.64752983, 0.57107152, 0.31789245, 0.39182499])

for i in range(n):
    out = dmp.step(start=demo_qs[:, 0].reshape(-1,1), goal=demo_qs[:,-1].reshape(-1,1))
    # out = dmp.step(start=demo_qs[:, 0].reshape(-1,1), goal=goal1.reshape(-1,1))
    # out = dmp.step(start=demo_qs[:, 0].reshape(-1,1), goal=goal2.reshape(-1,1))

    # out = dmp.step(start=demo_qs[:, 0].reshape(-1,1), goal=pour_wine_goal1.reshape(-1,1))
    # out = dmp.step(start=demo_qs[:, 0].reshape(-1,1), goal=pour_wine_goal2.reshape(-1,1))
    # out = dmp.step(start=demo_qs[:, 0].reshape(-1,1), goal=pour_wine_goal3.reshape(-1,1))
    q[:, i] = out[0].flatten()
    qd[:, i] = out[1].flatten()
    omega[:, i] = out[2].flatten()
    omegad[:, i] = out[3].flatten()

# np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'Imation_Quaternion.txt'), np.transpose(q))
# np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'Imation_Quaternion1.txt'), np.transpose(q))
# np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'Imation_Quaternion4.txt'), np.transpose(q))

# np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'pouring_wine_quaternion1.txt'), np.transpose(q))
# np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'pouring_wine_quaternion2.txt'), np.transpose(q))
# np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'pouring_wine_quaternion3.txt'), np.transpose(q))

plt.plot(demo_qs[0], label='x', color='b', linestyle='-')
plt.plot(q[0], color='b', linestyle='--')

plt.plot(demo_qs[1], label='y', color='r', linestyle='-')
plt.plot(q[1], color='r', linestyle='--')

plt.plot(demo_qs[2], label='z', color='g', linestyle='-')
plt.plot(q[2], color='g', linestyle='--')

plt.plot(demo_qs[3], label='w', color='m', linestyle='-')
plt.plot(q[3], color='m', linestyle='--')
plt.legend()
# plt.title('quat demo and dmp \nstart=' + str(demo_qs[:, 0]) + '\n end=' + str(demo_qs[:, -1]))
plt.show()
# plt.savefig(os.path.join('PouringWater2', 'Pictures', 'DynMovPrimitives_Imation_Quaternion.png'), dpi=100)
# plt.savefig(os.path.join('PouringWater2', 'Pictures', 'DynMovPrimitives_Generalized_Quaternion1.png'), dpi=100)
# plt.savefig(os.path.join('PouringWater2', 'Pictures', 'DynMovPrimitives_Generalized_Quaternion4.png'), dpi=100)

# also do euler conversion for more easily interpretable units
# demo_rots = R.from_quat(demo_qs.T)
# euler_demo = R.as_euler(demo_rots, 'XYZ').T
# dmp_rots = R.from_quat(q.T)
# euler_roll = R.as_euler(dmp_rots, 'XYZ').T

# plt.plot(euler_demo[0], label='euler x', color='b', linestyle='-')
# plt.plot(euler_roll[0], color='b', linestyle='--')
#
# plt.plot(euler_demo[1], label='euler y', color='r', linestyle='-')
# plt.plot(euler_roll[1], color='r', linestyle='--')
#
# plt.plot(euler_demo[2], label='euler z', color='g', linestyle='-')
# plt.plot(euler_roll[2], color='g', linestyle='--')
# plt.legend()
# plt.title('euler demo and dmp \n start=' + str(euler_demo[:,0])
#           + '\n end=' + str(euler_demo[:,-1]))
# plt.show()

# assume all rotations are relative to a global frame. We will be rotating
# a single point point at [0, 0, 1] on unit circle to plot the paths
# start = np.array([0,1,0])
# demo_path = demo_rots.apply(start)
# dmp_path = dmp_rots.apply(start)
# axis = omega[:, 0] / np.linalg.norm(omega[:, 0])
#
# # change rotation axis to world frame
# axis = R.inv(demo_rots[0]).apply(axis)
#
# # plotting paths on sphere
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# draw sphere
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = np.cos(u)*np.sin(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(v)
# ax.plot_wireframe(x, y, z, color="r", alpha=0.1)
# ax.plot(demo_path[:, 0], demo_path[:, 1], demo_path[:, 2], color='b')
# ax.plot(dmp_path[:, 0], dmp_path[:, 1], dmp_path[:, 2], color='orange')
# ax.plot([demo_path[0, 0]], [demo_path[0, 1]], [demo_path[0, 2]], marker='o')
# # ax.plot([axis[0]], [axis[1]], [axis[2]], marker='o')
# ax.plot([demo_path[-1, 0]], [demo_path[-1, 1]], [demo_path[-1, 2]], marker='o')
# plt.show()
