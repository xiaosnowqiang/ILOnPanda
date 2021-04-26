from scipy.spatial.transform import Rotation as R
import numpy as np
import os

cartesian_pose0 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_Position200_Orientation20.txt'))
cartesian_pose1 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_variableimpedance200_550-10_20.txt'))
cartesian_pose2 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_Position550_Orientation20.txt'))
generalized_position1 = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','generalized_position1.txt'))
generalized_quaternion1 = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','Imation_Quaternion1.txt'))

import matplotlib.pyplot as plt
labels = ['X','Y','Z']
for i in range(3):
    plt.plot(generalized_position1[i,:],lw=3,label='ref')
    plt.plot(cartesian_pose0[:11000,i+12],lw=1.5,label='imp200')
    plt.plot(cartesian_pose1[:11000,i+12],lw=1.5,label='var_imp')
    plt.plot(cartesian_pose2[:11000,i+12],lw=1.5,label='imp500')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join('PouringWater2', 'Pictures', 'VariablePositionControlIllustration'+labels[i]+'.png'), dpi=100)


cartesian_pose0 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_Position550_Orientation10.txt'))
cartesian_pose1 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_variableimpedance200_550-10_20.txt'))
cartesian_pose2 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_Position550_Orientation20.txt'))
cartesian_pose3 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_Position550_Orientation30.txt'))
#
#transform imation quaternion into so3 distance
cartesian_matrix0 = cartesian_pose0.reshape((-1,4,4))
cartesian_rotation_matrix0 = np.zeros((len(cartesian_matrix0),3,3))
for i in range(len(cartesian_matrix0)):
    cartesian_rotation_matrix0[i,:,:] = cartesian_matrix0[i,:3,:3].T

cartesian_quat0 = R.from_matrix(cartesian_rotation_matrix0)
cartesian_quat0 = cartesian_quat0.as_quat()

cartesian_matrix1 = cartesian_pose1.reshape((-1,4,4))
cartesian_rotation_matrix1 = np.zeros((len(cartesian_matrix1),3,3))
for i in range(len(cartesian_matrix1)):
    cartesian_rotation_matrix1[i,:,:] = cartesian_matrix1[i,:3,:3].T

cartesian_quat1 = R.from_matrix(cartesian_rotation_matrix1)
cartesian_quat1 = cartesian_quat1.as_quat()
#

cartesian_matrix2 = cartesian_pose2.reshape((-1,4,4))
cartesian_rotation_matrix2 = np.zeros((len(cartesian_matrix2),3,3))
for i in range(len(cartesian_matrix2)):
    cartesian_rotation_matrix2[i,:,:] = cartesian_matrix2[i,:3,:3].T

cartesian_quat2 = R.from_matrix(cartesian_rotation_matrix2)
cartesian_quat2 = cartesian_quat2.as_quat()

cartesian_matrix3 = cartesian_pose3.reshape((-1,4,4))
cartesian_rotation_matrix3 = np.zeros((len(cartesian_matrix3),3,3))
for i in range(len(cartesian_matrix3)):
    cartesian_rotation_matrix3[i,:,:] = cartesian_matrix3[i,:3,:3].T

cartesian_quat3 = R.from_matrix(cartesian_rotation_matrix3)
cartesian_quat3 = cartesian_quat3.as_quat()

for i in range(4):
    plt.plot(generalized_quaternion1[:,i],lw=3,label='ref')
    plt.plot(cartesian_quat0[:11000,i],lw=1.5,label='imp10')
    plt.plot(cartesian_quat1[:11000,i],lw=1.5,label='var_imp')
    plt.plot(cartesian_quat2[:11000,i],lw=1.5,label='imp20')
    plt.plot(cartesian_quat3[:11000,i],lw=1.5,label='imp30')
    plt.legend()
    plt.show()



