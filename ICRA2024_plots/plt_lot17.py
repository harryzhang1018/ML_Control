from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt

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

#ref_states = genfromtxt('/home/harry/Documents/test_plot/wpts_mpc/data/Circle_Traj.csv', delimiter=',')
#ref_states = genfromtxt('./pid_sin/sin_test_v1.csv', delimiter=',')
ref_states = genfromtxt('./paths/lot17_sinsquare_multi_vels.csv', delimiter=',')
#ref_states = genfromtxt('./pid_sin/sin_test_v5.csv', delimiter=',')
ref_x = ref_states[:,0]
ref_y = ref_states[:,1]



def data_read(filename):
    data = genfromtxt(filename, delimiter=',')
    return data


data_type_1 = '(NN Trained By MPC)'
data_type_2 = '(PD Trained By MPC)'
data_type_3 = '(NN Trained By Manual Control)'
data_type_4 = '(PD Trained By Manual Control)'
print("Select an option:")
print("A - NN Trained By MPC")
print("B - PD Trained By MPC")
print("C - NN Trained By Manual Control")
print("D - PD Trained By Manual Control")
data_type = input("Enter your choice (A/B/C/D): ").upper()
if data_type == "A":
    title_plot = data_type_1
elif data_type == "B":
    title_plot = data_type_2
elif data_type == "C":
    title_plot = data_type_3
elif data_type == "D":
    title_plot = data_type_4

foldername = './data/'
filename = 'ms_mc_bb_1.csv'
file = foldername + filename
data = data_read(file)
rx = data[:,0]
ry = data[:,1]
gtx = data[:,2]
#print('size of x outside function: ', gtx.shape)
gty = data[:,3]


gt_v = compute_speed_vector(gtx, gty)
EKFx = data[:,4]
EKFy = data[:,5]

plt.figure(figsize=(5,10))
plt.plot(ref_x,ref_y,'--',label='reference trajectory')
# plt.plot(rx,ry,label='gps')
plt.plot(gtx,gty,label='trajectory in reality using RTKGPS')
# plt.plot(EKFx,EKFy,label='state estimation')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('lot 17 '+ title_plot)
plt.tight_layout()
#plt.show()

show_sim_res = False
if show_sim_res:
    # sim_res = genfromtxt('./test_0710/simulation_sinsquare.csv', delimiter=',')
    sim_res = genfromtxt('./Simulation_ML/ml_mc_pd.csv', delimiter=',')
    x = sim_res[:,0]
    y = sim_res[:,1]
    sim_throttle = sim_res[:,6]
    sim_steering = sim_res[:,7]
    #plt.plot(ref_x,ref_y,'--',label='reference trajectory')
    plt.plot(x,y,label='trajectory in simulation')
    plt.legend(fontsize="12.5")

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

#time_sim = np.arange(sim_steering.shape[0])*0.1    
time_real = np.arange(data[:,9].shape[0])*0.1

plot_profile = False
if plot_profile:
    plt.figure(figsize=(10,3))
    plt.subplot(2,1,1)
    plt.plot(range(data[:,10].shape[0]),data[:,6],label='heading from imu')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(range(data[:,11].shape[0]),data[:,11],label='roll from imu')
    plt.legend()
    plt.figure(figsize=(10,3))
    plt.subplot(2,1,1)
    plt.title('control profile '+ title_plot)
    plt.plot(time_real,data[:,8],label='throttle')
    plt.plot(time_sim,sim_throttle,label='throttle in sim')
    plt.xlabel('time (s)')
    plt.legend(fontsize="12.5")
    plt.subplot(2,1,2)
    plt.plot(time_real,data[:,9],label='steering')
    plt.plot(time_sim,sim_steering,label='steering in sim')
    plt.xlabel('time (s)')
    plt.legend(fontsize="12.5")
    plt.tight_layout()
plt.tight_layout()
plt.savefig('image.png')