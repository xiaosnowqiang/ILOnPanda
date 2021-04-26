import numpy as np
import os
import matplotlib.pyplot as plt
# pose_error0 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_error_Position200_Orientation20.txt'))
# pose_error1 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_error_Position550_Orientation20.txt'))
# pose_error2 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_error_Position550_Orientation10.txt'))
# pose_error3 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_error_Position550_Orientation20.txt'))
# pose_error4 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_error_Position550_Orientation30.txt'))

# pose_error = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWaterFinalTest','cartesian_pose_error_variableimpedance200_550-10_20.txt'))
# for i in range(3):
#     plt.plot(pose_error0[:,i],lw=3,label='Imp200')
#     plt.plot(pose_error1[:,i],lw=3,label='Imp550')
#     plt.plot(pose_error[:,i],lw=3,label='VarImp')
#     plt.legend()
#     plt.show()
#
# for i in range(3,6):
#     plt.plot(pose_error2[:, i], lw=3, label='Imp10')
#     plt.plot(pose_error3[:, i], lw=3, label='Imp20')
#     plt.plot(pose_error4[:, i], lw=3, label='Imp30')
    # plt.plot(pose_error[:, i], lw=3, label='VarImp')
    # plt.legend()
    # plt.show()

wine_pose_error1 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWine','high_variable_impedance','cartesian_pose_error1.txt'))
wine_pose_error2 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWine','high_variable_impedance','cartesian_pose_error2.txt'))
wine_pose_error3 = np.loadtxt(os.path.join('PouringWater2','Recorded_data','PouringWine','high_variable_impedance','cartesian_pose_error3.txt'))

for i in range(6):
    plt.plot(wine_pose_error1[:,i],lw=3,label='1')
    plt.plot(wine_pose_error2[:,i],lw=3,label='2')
    plt.plot(wine_pose_error3[:,i],lw=3,label='3')
    plt.legend()
    plt.show()
