from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt
from operator import itemgetter


#using the refx and refy, want to find the first point that and the throttle and steering for that spot.
def preprocess_data(refx, refy, data, datat, datas):
    x = data[:,0]
    y = data[:,1]
    throttle = datat
    steering = datas
    new_t = []
    new_s = []
    my_ref = []
    my_data = []
    for i in range(0, len(data)):
        my_x = x[i]
        my_y = y[i]
        best_r = 0
        best_d = np.sqrt((my_x-refx[0])**2+(my_y-refy[0])**2)
        for j in range(1,len(refx)):
            d = np.sqrt((my_x-refx[j])**2+(my_y-refy[j])**2)
            if(d<best_d):
                best_r = j
                best_d = d
        if(not best_r in my_ref):
            my_ref.append(best_r)
            new_t.append(throttle[i])
            new_s.append(steering[i])

            my_data.append(np.array([best_r, throttle[i], steering[i]]))
    my_data = sorted(my_data, key=itemgetter(0))
    my_ref = []
    new_t = []
    new_s = []
    for i in my_data:
        my_ref.append(i[0])
        new_t.append(i[1])
        new_s.append(i[2])



    return my_ref, new_t, new_s

def data_read(filename):
    data = genfromtxt(filename, delimiter=',')
    return data

def compute_speed_vector(gtx, gty):
    #print('size of x inside function: ', gtx.shape)
    # Compute the differences between consecutive elements
    dx = np.diff(gtx,n=1)*10
    dy = np.diff(gty,n=1)*10
    #print(dx.shape)
    # Compute the speed vector
    speed = np.sqrt(dx**2 + dy**2)
    speed = np.insert(speed,0,0)
    for i in range(speed.shape[0]-1):
        if (abs(speed[i+1]-speed[i])>0.4):
            speed[i+1] = speed[i]
    return speed

#The path
ref_states = genfromtxt('./paths/lot17_sinsquare_multi_vels.csv', delimiter=',')
ref_x = ref_states[:,0]
ref_y = ref_states[:,1]


data_type_1 = '(NN Trained By MPC)'
data_type_3 = '(NN Trained By Manual Control)'

title_plot = data_type_1

foldername = './data/'
#manual control single speed
# filename = 'mc_bb_1.csv'
# title_plot = '(NN Trained By Manual Control, single speed)'

#mpc single speed
filename = 'mpc_bb_2.csv'
title_plot = '(NN Trained By MPC, single speed)'

#multispeed mc
#filename = 'ms_mc_bb_1.csv'
#title_plot = '(NN Trained By MPC, multi speed)'

#multispeed mpc
#filename = 'ms_mpc_bb_1.csv'
##title_plot = '(NN Trained By Manual Control, multi speed)'
file = foldername + filename
data = data_read(file)
rx = data[:,0]
ry = data[:,1]
gtx = data[:,2]
gty = data[:,3]

gt_v = compute_speed_vector(gtx, gty)
EKFx = data[:,4]
EKFy = data[:,5]

plt.figure(figsize=(5,10))
plt.plot(ref_x,ref_y,'--',label='reference trajectory')
# plt.plot(rx,ry,label='gps')
plt.plot(gtx,gty,label='trajectory in reality')
# plt.plot(EKFx,EKFy,label='state estimation')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('lot 17 '+ title_plot, fontsize="12")
plt.tight_layout()
#plt.show()

show_sim_res = True
if show_sim_res:
    # sim_res = genfromtxt('./test_0710/simulation_sinsquare.csv', delimiter=',')
    # sim_res = genfromtxt('./data/keras_ml_learMC_1la_0808_sim.csv', delimiter=',')

    #mpc single speed
    sim_res = genfromtxt('./data/keras_ml_learMPC_1la_0808_sim.csv', delimiter=',')

    #mc multispeed
    # sim_res = genfromtxt('./data/keras_ml_learMC_1la_0828_sim_30v.csv', delimiter=',')

    #mpc multispeed
    # sim_res = genfromtxt('./data/keras_ml_learMPC_1la_0828_sim_30v.csv', delimiter=',')

    x = sim_res[:,0]
    y = sim_res[:,1]
    sim_throttle = sim_res[:,6]
    sim_steering = sim_res[:,7]
    #plt.plot(ref_x,ref_y,'--',label='reference trajectory')
    plt.plot(x,y,label='trajectory in simulation')
    plt.legend(fontsize="16")

time_sim = np.arange(sim_steering.shape[0])*0.1    
time_real = np.arange(data[:,9].shape[0])*0.1

my_ref_sim, sim_t, sim_s = preprocess_data(ref_x, ref_y, sim_res, sim_res[:,6], sim_res[:,7])
my_ref_r, r_t, r_s = preprocess_data(ref_x, ref_y, data, data[:,8], data[:,9])

plot_profile = True
if plot_profile:
    plt.figure(figsize=(10,3))
    # plt.subplot(2,1,1)
    # plt.plot(range(data[:,10].shape[0]),data[:,6],label='heading from imu')
    # plt.legend()
    # plt.subplot(2,1,2)
    # plt.plot(range(data[:,11].shape[0]),data[:,11],label='roll from imu')
    # plt.legend()
    # plt.figure(figsize=(10,3))
    plt.subplot(2,1,1)
    plt.title('control profile '+ title_plot)
    plt.plot(my_ref_r,r_t,label='throttle')
    plt.plot(my_ref_sim,sim_t,label='throttle in sim')
    plt.xlabel('time (s)')
    plt.legend(fontsize="12.5")
    plt.xlim([-300,1100])
    plt.subplot(2,1,2)
    plt.plot(my_ref_r,r_s,label='steering')
    plt.plot(my_ref_sim,sim_s,label='steering in sim')
    plt.xlabel('time (s)')
    plt.legend(fontsize="12.5")
    plt.xlim([-300,1100])
    plt.tight_layout()
plt.tight_layout()
plt.savefig('image.png')
