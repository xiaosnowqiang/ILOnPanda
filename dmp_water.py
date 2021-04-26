import numpy as np
from Pos_DMP import DMP_Agent
from Quat_DMP import QuatDMP_Agent
import matplotlib.pyplot as plt

def Variance2Impedance(variances,min_impedance,max_impedance):
    #transforme variances into impedances
    #variances : [timesteps,3]
    #we transform position and orientation impedance separately

    min_var = np.min(variances)
    max_var = np.max(variances)
    # in a(x-max_var)^2+min_impedance 
    a = (max_impedance - min_impedance) / (min_var - max_var)**2
    impedances = a * (variances - max_var)**2 + min_impedance
    return impedances 


if __name__ == '__main__':
    import os
    #define min_position_impedance, max_position_impedance, min_orien_impedance, max_orien_impedance
    min_position_impedance, max_position_impedance = 200,550
    min_orien_impedance, max_orien_impedance = 10,20

    # read refrence trajectorries and variances
    gmm_reference_position = np.loadtxt(os.path.join('PouringWater2', 'FinedTrajectories', 'gmm_reference_position.txt')).T
    gmm_position_variance = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','gmm_position_variance.txt')).T
    gmm_reference_quaternion = np.loadtxt(os.path.join('PouringWater2', 'FinedTrajectories', 'gmm_reference_quaternion.txt')).T
    gmm_quat_dist_var = np.loadtxt(os.path.join('PouringWater2', 'FinedTrajectories', 'gmm_quaternion_distance_variance.txt')).T

    pose_y_ref = np.vstack((gmm_reference_position, gmm_reference_quaternion))
    np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'reference_pose.txt'), pose_y_ref)
    # translate variances into impedances
    position_impedances = Variance2Impedance(gmm_position_variance,min_position_impedance,max_position_impedance)
    orientation_impedances = Variance2Impedance(gmm_quat_dist_var,min_orien_impedance,max_orien_impedance)
    
    # set initial pose and target pose
    position_y0 = gmm_reference_position[:,0]
    orientation_y0 = gmm_reference_quaternion[:,0]
    position_goal0 = gmm_reference_position[:,-1]
    orientatioin_goal0 = gmm_reference_quaternion[:,-1]
    ## goal generalization
    # 左上角末端位置：位置反向，姿态正向
    pose_goal1 = [0.795716,-0.129779,0.560809,0.59267273,0.59229562,0.28305234,0.46669724]
    # 左上角末端位置：位置反向，姿态反向
    # pose_goal2 = [0.800356,0.0614152,0.585289,-0.54275816,0.54360883, -0.33089064,  0.54810073]
    # 右下角末端位置：位置正向，姿态反向
    # pose_goal3 = [0.575443,0.520559,0.580677, 0.73848266, -0.40057892,  0.13216472, -0.52603458]
    # 右下角末端位置：位置正向，姿态正向
    pose_goal4 = [0.706232,0.376537,0.585502,0.24751033,0.71336469,0.50727476,0.41535741]
    print('\nposition_y0: ',position_y0)
    print('\norientation_y0: ',orientation_y0)
    print('\nposition_goal0: ',position_goal0)
    print('\norientation_goal0: ',orientatioin_goal0)
    #define a dmp_agent with 6 dmps
    dmp_agent_position = DMP_Agent(n_dmps=3,n_bfs=400,timesteps=11000,dt=.001,y0=position_y0,goal=position_goal0)
    position_y_des = gmm_reference_position
    #train 6 dmps and get weights [n_dmps,n_rbf]
    position_weights = dmp_agent_position.imitate_path(position_y_des)
    ##pose generalizetion
    position0, position_vel0, position_acc0 = dmp_agent_position.rollout()

    position_goal_adj = position_goal0
    # position_goal_adj[0] = position_goal0[0] - 0.018
    # position_adj,position_vel_adj,position_acc_Adj = dmp_agent_position.rollout(goal=position_goal_adj)
    position1, position_vel1, position_acc1 = dmp_agent_position.rollout(goal=pose_goal1[:3])
    position4, position_vel4, position_acc4 = dmp_agent_position.rollout(goal=pose_goal4[:3])
    np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'Imation_position.txt'), np.transpose(position0))
    # np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'adjust_position.txt'), np.transpose(position_adj))
    np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'generalized_position1.txt'), np.transpose(position1))
    np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'generalized_position4.txt'), np.transpose(position4))


    #create 6 more dmps for getting impedances weights
    # set initial impedance and target impedance
    impedance_y0 = np.hstack((position_impedances[:,0],orientation_impedances[:,0]))
    impedance_goal0 = np.hstack((position_impedances[:,-1],orientation_impedances[:,-1]))
    print('\nimpedance_y0:',impedance_y0)
    print('\nimpedancce_goal:', impedance_goal0)
    # define a dmp_agent with 6 dmps
    dmp_agent_impedance = DMP_Agent(n_dmps=6,n_bfs=2000,timesteps=11000,dt=.001,y0=impedance_y0,goal=impedance_goal0)
    impedance_y_ref = np.vstack((position_impedances,orientation_impedances))
    np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'reference_impedances.txt'), impedance_y_ref)
    # train 6 dmps and get weights [n_dmps,n_rbf]
    impedance_weights = dmp_agent_impedance.imitate_path(impedance_y_ref)
    # impedance_force_terms = dmp_agent_impedance.gen_force_term()
    impedance0, impedance_vel0, impedance_acc0= dmp_agent_impedance.rollout()
    # ##impedance generalization
    # impedance_goal4 = [1000,1000,1000,30,30,30]
    # impedance0, impedance_vel0, impedance_acc0 = dmp_agent_impedance.rollout(goal=impedance_goal0)
    # impedance4, impedance_vel4, impedance_acc4 = dmp_agent_impedance.rollout(goal=impedance_goal4)
    np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'generalized_impedance0.txt'), np.transpose(impedance0))
    # np.savetxt(os.path.join('PouringWater2', 'FinedTrajectories', 'generalized_impedance4.txt'), np.transpose(impedance4))

    indices = np.linspace(0, 11000, 11000).astype(int) * 0.001

    # plt.figure(figsize=(18, 10))
    # lables = ['RefX', 'RefY', 'RefZ', 'PreX', 'PreY', 'PreZ']
    # for i in range(3):
    #     plt.plot(indices, position_y_des[i, :], lw=3, label=lables[i])
    #     plt.plot(indices, position0[:, i], lw=3, label=lables[i + 3])
    # plt.xticks(fontsize=36)
    # plt.yticks(fontsize=36)
    # plt.xlabel('Time  [s]', fontsize=36)
    # plt.ylabel('Position [m]', fontsize=36)
    # plt.legend(loc='lower right', fontsize=18,)
    # plt.savefig(os.path.join('PouringWater2', 'Pictures', 'DynMovPrimitives_Reference_Position.png'), dpi=100)
    # plt.show()
    # #
    # plt.figure(figsize=(18, 10))
    # lables = ['RefX', 'RefY', 'RefZ', 'GenX', 'GenY', 'GenZ']
    # for i in range(3):
    #     plt.plot(indices, position_y_des[i, :], lw=3, label=lables[i])
    #     plt.plot(indices, position1[:, i], lw=3, label=lables[i + 3])
    # plt.xticks(fontsize=36)
    # plt.yticks(fontsize=36)
    # plt.xlabel('Time  [s]', fontsize=36)
    # plt.ylabel('Position [m]', fontsize=36)
    # plt.legend(loc='lower right', fontsize=18, )
    # plt.savefig(os.path.join('PouringWater2', 'Pictures', 'DynMovPrimitives_Generalized_Position1.png'), dpi=100)
    # plt.show()
    #
    # plt.figure(figsize=(18, 10))
    # lables = ['RefX', 'RefY', 'RefZ', 'PreX', 'PreY', 'PreZ']
    # for i in range(3):
    #     plt.plot(indices, impedance_y_ref[i, :], lw=3, label=lables[i])
    #     plt.plot(indices, impedance0[:, i], lw=3, label=lables[i + 3])
    # plt.xticks(fontsize=36)
    # plt.yticks(fontsize=36)
    # plt.xlabel('Time  [s]', fontsize=36)
    # plt.ylabel('Stiffness In Position', fontsize=36)
    # plt.legend(loc='lower right', fontsize=18)
    # plt.savefig(os.path.join('PouringWater2', 'Pictures', 'DynMovPrimitives_Reference_Position_Impedance.png'), dpi=100)
    # plt.show()
    #
    # plt.figure(figsize=(18, 10))
    # lables = ['RefX', 'RefY', 'RefZ', 'PreX', 'PreY', 'PreZ']
    # for i in range(3):
    #     plt.plot(indices, impedance_y_ref[i + 3, :], lw=3, label=lables[i])
    #     plt.plot(indices, impedance0[:, i + 3], lw=3, label=lables[i + 3])
    # plt.xticks(fontsize=36)
    # plt.yticks(fontsize=36)
    # plt.xlabel('Time  [s]', fontsize=36)
    # plt.ylabel('Stiffness In Orientation', fontsize=36)
    # plt.legend(loc='lower right', fontsize=18)
    # plt.savefig(os.path.join('PouringWater2', 'Pictures', 'DynMovPrimitives_Reference_Quaternion_Impedance.png'), dpi=100)
    # plt.show()