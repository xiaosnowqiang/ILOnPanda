import numpy as np
import os

#concantate position and orientation
pour_wine_position1 = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_position1.txt'))
pour_wine_quaternion1 = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_quaternion1.txt')).T

pour_wine_position2 = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_position2.txt'))
pour_wine_quaternion2 = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_quaternion2.txt')).T

pour_wine_position3 = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_position3.txt'))
pour_wine_quaternion3 = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_quaternion3.txt')).T

pouring_wine_pose1 = np.vstack((pour_wine_position1,pour_wine_quaternion1))
pouring_wine_pose2 = np.vstack((pour_wine_position2,pour_wine_quaternion2))
pouring_wine_pose3 = np.vstack((pour_wine_position3,pour_wine_quaternion3))

np.savetxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_pose1.txt'),pouring_wine_pose1)
np.savetxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_pose2.txt'),pouring_wine_pose2)
np.savetxt(os.path.join('PouringWater2','FinedTrajectories','pouring_wine_pose3.txt'),pouring_wine_pose3)

imation_pose = np.loadtxt(os.path.join('PouringWater2','FinedTrajectories','imation_pose.txt'))

import matplotlib.pyplot as plt
for i in range(7):
    plt.plot(imation_pose[i,:],label='imation')
    plt.plot(pouring_wine_pose3[i,:],label='imation')
    plt.legend()
    plt.show()