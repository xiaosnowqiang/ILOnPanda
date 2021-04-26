import sys
root_path = '/home/yan/Desktop/ILOnPanda'
sys.path.append(root_path)

import os
import numpy as np
import matplotlib.pyplot as plt
from gmr import GMM, plot_error_ellipses

NUM = 8
N_COMPONENTS = 5
RANDOM_STATE = 0
colors=["r", "g", "b", "y", "cyan"]


def gen_comp(pose,num,index):
    com = []
    for i in range(num):
        for j in range(len(pose[i])):
            com.append([pose[i][j,0],pose[i][j,index]])
    return np.array(com)

def regression(X_test,dataset,colors,name):
    gmm_position = GMM(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    gmm_position.from_samples(dataset, n_iter=100)
    predict_position, variance_position = gmm_position.predict(np.array([0]), X_test[:, np.newaxis])

    plt.figure(figsize=(18, 10))
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plot_error_ellipses(plt.gca(), gmm_position, colors=colors)
    plt.plot(X_test, predict_position.ravel(), c="k", lw=2)
    plt.xticks(fontsize=36, fontweight='bold')
    plt.yticks(fontsize=36, fontweight='bold')
    plt.xlabel('Time  [s]', fontsize=36)
    plt.ylabel('{}  [m]'.format(name), fontsize=36)
    plt.savefig(os.path.join('PouringWater2', 'Pictures', name+'.png'), dpi=100)

    return predict_position,variance_position

if __name__ == '__main__':
    pose = np.load(os.path.join('../Demonstrations', 'GMRInputs.npy'), allow_pickle=True)

    position_x, position_y, position_z = gen_comp(pose, NUM, 1), gen_comp(pose, NUM, 2), gen_comp(pose, NUM, 3)
    quaternion_x, quaternion_y, quaternion_z = gen_comp(pose, NUM, 4), gen_comp(pose, NUM, 4), gen_comp(pose, NUM, 6)

    #test indices
    X_test = np.linspace(0,11,11000)
    predict_position_x,variance_position_x = regression(X_test,position_x,colors,name='PositionX')
    predict_position_y,variance_position_y = regression(X_test, position_y, colors, name='PositionY')
    predict_position_z,variance_position_z = regression(X_test, position_z, colors, name='PositionZ')
    predict_quat_x, variance_quat_x = regression(X_test, quaternion_x, colors, name='QuaternionX')
    predict_quat_y, variance_quat_y = regression(X_test, quaternion_y, colors, name='QuaternionY')
    predict_quat_z, variance_quat_z = regression(X_test, quaternion_z, colors, name='QuaternionZ')

    ref_quat_dist = np.hstack((predict_quat_x, predict_quat_y, predict_quat_z))
    ref_quat = np.zeros((len(ref_quat_dist),4))
    for i in range(len(ref_quat)):
        ref_quat[i,0] = np.cos(np.linalg.norm(ref_quat_dist[i,:])/2)
        scale = np.sqrt(1-ref_quat[i,0]**2)
        ref_quat[i,1:] = scale * ref_quat_dist[i,:]/(np.linalg.norm(ref_quat_dist[i,:]))

    np.save(os.path.join('PouringWater2', 'FinedTrajectories', 'gmr_ref_pos.npy'),
            np.hstack((predict_position_x, predict_position_y, predict_position_z)))
    np.save(os.path.join('PouringWater2', 'FinedTrajectories', 'gmr_pos_var.npy'),
            np.hstack((variance_position_x, variance_position_y, variance_position_z)))
    np.save(os.path.join('PouringWater2', 'FinedTrajectories', 'gmm_ref_quat.npy'),
               ref_quat)
    np.save(os.path.join('PouringWater2', 'FinedTrajectories', 'gmm_quat_dis_var.npy'),
               np.hstack((variance_quat_x, variance_quat_y, variance_quat_z)))

    #plot positon x variance
    plt.figure(figsize=(18, 10))
    plt.plot(X_test,np.sqrt(variance_position_x.ravel()),c='red',lw=3,label='PositionX')
    plt.plot(X_test,np.sqrt(variance_position_y.ravel()),c='green',lw=3,label='PositionY')
    plt.plot(X_test,np.sqrt(variance_position_z.ravel()),c='blue',lw=3,label='PositionZ')
    plt.xticks(fontsize=36,fontweight='bold')
    plt.yticks(fontsize=36,fontweight='bold')
    plt.xlabel('Time  [s]',fontsize=36)
    plt.ylabel('Standard Deviation In Position',fontsize=36)
    plt.legend(loc='upper right',fontsize=24)
    plt.savefig(os.path.join('PouringWater2', 'Pictures', 'VarianceInPosition.png'), dpi=100)


    #plot quaternion x variance
    plt.figure(figsize=(18, 10))
    plt.plot(X_test,np.sqrt(variance_quat_x.ravel()),c='red',lw=3,label='QuaternionX')
    plt.plot(X_test,np.sqrt(variance_quat_y.ravel()),c='green',lw=3,label='QuaternionY')
    plt.plot(X_test,np.sqrt(variance_quat_z.ravel()),c='blue',lw=3,label='QuaternionZ')
    plt.xticks(fontsize=36,fontweight='bold')
    plt.yticks(fontsize=36,fontweight='bold')
    plt.xlabel('Time  [s]',fontsize=36)
    plt.ylabel('Standard Deviation In Quaternion',fontsize=36)
    plt.legend(loc='upper right',fontsize=24)
    plt.savefig(os.path.join('PouringWater2', 'Pictures', 'VarianceInQuaternion.png'), dpi=100)
