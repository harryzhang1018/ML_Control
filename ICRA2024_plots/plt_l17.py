from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt

# Add matplotlib rc parameters to beautify plots

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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
filename = 'mc_bb_1.csv'
title_plot = '(NN Trained By Manual Control, single speed)'

#mpc single speed
# filename = 'mpc_bb_2.csv'
# title_plot = '(NN Trained By MPC, single speed)'

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
    # sim_res = genfromtxt('./data/keras_ml_learMPC_1la_0808_sim.csv', delimiter=',')

    #mc multispeed
    sim_res = genfromtxt('./data/keras_ml_learMC_1la_0828_sim_30v.csv', delimiter=',')

    #mpc multispeed
    # sim_res = genfromtxt('./data/keras_ml_learMPC_1la_0828_sim_30v.csv', delimiter=',')

    x = sim_res[:,0]
    y = sim_res[:,1]
    sim_throttle = sim_res[:,6]
    sim_steering = sim_res[:,7]
    #plt.plot(ref_x,ref_y,'--',label='reference trajectory')
    plt.plot(x,y,label='trajectory in simulation')
    plt.legend(fontsize="16")

plot_speed_map = False
if plot_speed_map:
    plt.figure(figsize=(5,10))
    # plt.subplot(1,2,1)
    # plt.scatter(x, y, c=sim_res[:,3], cmap='jet')
    # plt.colorbar(label='Velocity Magnitude') 
    # plt.title('Simulation Speed Map')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    #plt.subplot(1,2,2)
    plt.scatter(gtx, gty, c=gt_v, cmap='jet')
    plt.colorbar(label='Velocity Magnitude') 
    plt.title('Reality Speed Map')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

time_sim = np.arange(sim_steering.shape[0])*0.1    
time_real = np.arange(data[:,9].shape[0])*0.1

plot_profile = True
if plot_profile:
    plt.figure(figsize=(10,3))
    plt.subplot(2,1,1)
    plt.plot(range(data[:,10].shape[0]),data[:,6],label='heading from imu')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(range(data[:,11].shape[0]),data[:,11],label='roll from imu')
    plt.legend()
    # Set up the figure
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (10,6), sharex = True)
    # Set up the title
    fig.suptitle('Control Profile '+ title_plot)
    # Plot row 1
    axes[0].plot(time_real,data[:,8],label='Reality')
    axes[0].plot(time_sim,sim_throttle,label='Simualtion')
    axes[0].set_ylabel('Throttle')
    # Need only 1 legend
    axes[0].legend()
    # Plot row 2
    axes[1].plot(time_real,data[:,9],label='Reality')
    axes[1].plot(time_sim,sim_steering,label='Simulation')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Steering')
    fig.tight_layout()

# Save both as eps and png
plt.savefig('image.eps', format='eps', dpi = 3000) # Use eps for latex as it is vectorized
plt.savefig('image.png', facecolor='w', format='png', dpi = 1000) # Use png for other purposes