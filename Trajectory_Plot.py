from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt

ref_states_sin = genfromtxt('./data/Sin_Traj.csv', delimiter=',')
ref_sin_x = ref_states_sin[:,0]
ref_sin_y = ref_states_sin[:,1]

ref_states_circle = genfromtxt('./data/Circle_Traj_CW.csv', delimiter=',')
ref_circle_x = ref_states_circle[:,0]
ref_circle_y = ref_states_circle[:,1]

train_set_1 = genfromtxt('./data/mpc_sin_train1.csv', delimiter=',')
x_t1 = train_set_1[:,0]
y_t1 = train_set_1[:,1]

train_set_2 = genfromtxt('./data/mpc_sin_train2.csv', delimiter=',')
x_t2 = train_set_2[:,0]
y_t2 = train_set_2[:,1]

train_set_3 = genfromtxt('./data/mpc_sin_train3.csv', delimiter=',')
x_t3 = train_set_3[:,0]
y_t3 = train_set_3[:,1]

ml_set_1 = genfromtxt('./data/mpc_sin_ml1.csv', delimiter=',')
x_ml_1 = ml_set_1[:,0]
y_ml_1 = ml_set_1[:,1]

ml_set_2 = genfromtxt('./data/circle_ml1.csv', delimiter=',')
x_ml_2 = ml_set_2[:,0]
y_ml_2 = ml_set_2[:,1]

ekf_t1 = genfromtxt('./data/ekf_sin_t3.csv',delimiter=',')
gps_x_ekf_t1 = ekf_t1[:,0]
gps_y_ekf_t1 = ekf_t1[:,1]
gt_x_ekf_t1 = ekf_t1[:,2]
gt_y_ekf_t1 = ekf_t1[:,3]
# # plot the mpc controlled trajectory 
# plt.figure(figsize=(10,5))
# plt.plot(ref_sin_x,ref_sin_y,'*-',label='reference trajectory')
# plt.plot(x_t1,y_t1,label='train set 1: la = 0.75')
# plt.plot(x_t2,y_t2,label='train set 2: la = 1.0')
# plt.plot(x_t3,y_t3,label='train set 3: la = 0.7')
# plt.plot(x_ml_1,y_ml_1,label='ML Controller: la = 0.7',linewidth=2)
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,10))
# plt.plot(ref_circle_x,ref_circle_y,'*-',label='reference trajectory')
# plt.plot(x_ml_2,y_ml_2,label='ML Controller: la = 0.7',linewidth=2)
# plt.legend()
# plt.show()
plt.figure(figsize=(10,5))
plt.plot(gps_x_ekf_t1,gps_y_ekf_t1,label='measurement')
plt.plot(gt_x_ekf_t1,gt_y_ekf_t1,label='ground truth')
plt.legend()
plt.show()