import sys
root_path = '/home/yan/Desktop/ILOnPanda'
sys.path.append(root_path)

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

demos_root_path = os.path.join(os.path.abspath(os.curdir), 'Demonstrations')
init_time = np.array([1250,300,500,1100,1000,1,300,1000])
final_time = np.array([7000,8000,10000,8000,10000,8100,7500,9500])
length = final_time - init_time
time_scale = 10000 / (final_time - init_time)
NUM = 8

def load_traj(num):
    assert num > 0
    cartesian_pose_demos = []
    for i in range(num):
        demo_pose_path = os.path.join(demos_root_path,'cartesian_pose{i}.txt'.format(i=i))
        demo_pose = []
        with open(demo_pose_path, 'r') as demo:
            while True:
                lines = demo.readline()
                if not lines:
                    break
                r_list = [float(item) for item in lines.split()]
                demo_pose.append(np.transpose(np.reshape(r_list,(4,4))))

        cartesian_pose_demos.append(demo_pose)
    return np.array(cartesian_pose_demos)

def Rotation2Quat(num):
    assert num > 0
    cartesian_quaternion = []
    for i in range(8):
        tempr = R.from_matrix(cartesian_rotation_matrix[i,:,:,:])
        quaternion = tempr.as_quat()
        cartesian_quaternion.append(quaternion)
    return np.array(cartesian_quaternion)

def align_traj(num):
    assert num > 0
    cartesian_pose = []
    for i in range(8):
        cartesian_pos_temp = cartesian_position[i,init_time[i]:final_time[i],:]
        cartesian_quat_temp = cartesian_quaternion[i,init_time[i]:final_time[i],:]
        indice = np.linspace(0,10,length[i]).reshape((-1,1))
        cartesian_pose_temp = np.hstack((indice,cartesian_pos_temp,cartesian_quat_temp))
        cartesian_pose.append(cartesian_pose_temp)
    return cartesian_pose

#calculate quaternion distance in base frame
def quat_log(cartesian_pos,num):
    quaternion_distance = []
    for i in range(num):
        temp_v = np.arccos(cartesian_pose[i][:,7])
        temp_u = 1 / np.linalg.norm(cartesian_pose[i][:,4:7],axis=1)
        for j in range(len(cartesian_pose[i])):
            cartesian_pose[i][j,4:7] = 2 * temp_v[j] * temp_u[j] * cartesian_pose[i][j,4:7]
        quaternion_distance.append(cartesian_pose[i][:,4:7])
    return quaternion_distance


if __name__ == '__main__':
    #cartesian matrix
    cartesian_pose_demos = load_traj(NUM)
    cartesian_position = cartesian_pose_demos[:,:,:3,3]
    cartesian_rotation_matrix = cartesian_pose_demos[:,:,:3,:3]
    cartesian_quaternion = Rotation2Quat(NUM)
    cartesian_pose = align_traj(NUM)

    quat_dis = quat_log(cartesian_pose,NUM)
    rest_indice = np.linspace(10, 11, 1000)

    GMRInputs = []
    for i in range(NUM):
        temp = np.ones((1000, 7))
        temp[:, 0] = rest_indice
        Transformed_temp = np.hstack(
            (cartesian_pose[i][:, :4], quat_dis[i]))
        temp[:, 1:] = temp[:, 1:] * Transformed_temp[-1, 1:]
        Transformed_temp = np.vstack((Transformed_temp, temp))
        GMRInputs.append(Transformed_temp)

    np.save(os.path.join('Demonstrations', 'aligned_trajectories.npy'), cartesian_pose)
    np.save(os.path.join('Demonstrations', 'GMRInputs.npy'), GMRInputs)
