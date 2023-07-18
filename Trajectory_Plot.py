from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt

sin_path = './data/sin'
circle_path = './data/circle'

def plot(ref_path):
    ref_states = genfromtxt(ref_path+'_traj.csv', delimiter=',')
    ref_x = ref_states[:,2]
    ref_y = ref_states[:,3]

    training_data_1 = genfromtxt(ref_path+'_sim_testing_1.csv',delimiter=',')
    gtx_1 = training_data_1[:,0]
    gty_1 = training_data_1[:,1]

    training_data_2 = genfromtxt(ref_path+'_sim_testing_2.csv',delimiter=',')
    gtx_2 = training_data_2[:,0]
    gty_2 = training_data_2[:,1]

    training_data_3 = genfromtxt(ref_path+'_sim_testing_3.csv',delimiter=',')
    gtx_3 = training_data_3[:,0]
    gty_3 = training_data_3[:,1]

    training_data_4 = genfromtxt(ref_path+'_sim_testing_4.csv',delimiter=',')
    gtx_4 = training_data_4[:,0]
    gty_4 = training_data_4[:,1]

    training_data_5 = genfromtxt(ref_path+'_sim_testing_5.csv',delimiter=',')
    gtx_5 = training_data_5[:,0]
    gty_5 = training_data_5[:,1]

    ml_coef_control_data = genfromtxt(ref_path+'_sim_ml_coef.csv',delimiter=',')
    ml_x_coef = ml_coef_control_data[:,0]
    ml_y_coef = ml_coef_control_data[:,1]

    ml_bb_control_data = genfromtxt(ref_path+'_sim_ml_bb.csv',delimiter=',')
    ml_x_bb = ml_bb_control_data[:,0]
    ml_y_bb = ml_bb_control_data[:,1]

    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(ref_x,ref_y,'--',label='reference path')
    plt.plot(gtx_1,gty_1,label='harry control run 1')
    plt.plot(gtx_2,gty_2,label='harry control run 2')
    plt.plot(gtx_3,gty_3,label='harry control run 3')
    plt.plot(gtx_4,gty_4,label='harry control run 4')
    plt.plot(gtx_5,gty_5,label='harry control run 5')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(ref_x,ref_y,'--',label='reference path')
    plt.plot(ml_x_coef,ml_y_coef,label = 'Imitation Learning (coefficient)')
    plt.plot(ml_x_bb,ml_y_bb,label = 'Imitation Learning (black box)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
plot(circle_path)