from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt

ref_states_sin = genfromtxt('./data/sin_test_v1.csv', delimiter=',')
ref_sin_x = ref_states_sin[:,0]
ref_sin_y = ref_states_sin[:,1]

training_data_1 = genfromtxt('./data/sin_sim_testing_1.csv',delimiter=',')
gtx_1 = training_data_1[:,0]
gty_1 = training_data_1[:,1]

training_data_2 = genfromtxt('./data/sin_sim_testing_2.csv',delimiter=',')
gtx_2 = training_data_2[:,0]
gty_2 = training_data_2[:,1]

training_data_3 = genfromtxt('./data/sin_sim_testing_3.csv',delimiter=',')
gtx_3 = training_data_3[:,0]
gty_3 = training_data_3[:,1]

training_data_4 = genfromtxt('./data/sin_sim_testing_4.csv',delimiter=',')
gtx_4 = training_data_4[:,0]
gty_4 = training_data_4[:,1]

training_data_5 = genfromtxt('./data/sin_sim_testing_5.csv',delimiter=',')
gtx_5 = training_data_5[:,0]
gty_5 = training_data_5[:,1]

ml_coef_control_data = genfromtxt('./data/sin_sim_ml_coef.csv',delimiter=',')
ml_x_coef = ml_coef_control_data[:,0]
ml_y_coef = ml_coef_control_data[:,1]

ml_bb_control_data = genfromtxt('./data/sin_sim_ml_bb.csv',delimiter=',')
ml_x_bb = ml_bb_control_data[:,0]
ml_y_bb = ml_bb_control_data[:,1]
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
plt.figure(figsize=(15,7))
plt.subplot(2,1,1)
plt.plot(ref_sin_x,ref_sin_y,'--',label='reference path')
plt.plot(gtx_1,gty_1,label='harry control run 1')
plt.plot(gtx_2,gty_2,label='harry control run 2')
plt.plot(gtx_3,gty_3,label='harry control run 3')
plt.plot(gtx_4,gty_4,label='harry control run 4')
plt.plot(gtx_5,gty_5,label='harry control run 5')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.subplot(2,1,2)
plt.plot(ref_sin_x,ref_sin_y,'--',label='reference path')
plt.plot(ml_x_coef,ml_y_coef,label = 'Imitation Learning (coefficient)')
plt.plot(ml_x_bb,ml_y_bb,label = 'Imitation Learning (black box)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.tight_layout()
plt.show()