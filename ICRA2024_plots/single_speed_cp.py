#CONTROL PROFILE

from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.signal

traj = np.loadtxt('./data/mc_bb_2.csv', delimiter=',')
#path = genfromtxt('/home/scaldararu/Documents/Python_Data_Manipulation/test_vel/path.csv', delimiter=',')
path = np.loadtxt('./paths/rect.csv', delimiter=',')
x = traj[:,0]
y = traj[:,1]
px = path[:,0]
py = path[:,1]
pv = path[:,3]
throttle = traj[:,6]
steering = traj[:,7]
tv = [0]
i = [0]
k = []

for j in range(1,y.size):
    i.append(j)
    tv.append(np.sqrt((x[j]-x[j-1])**2+(y[j]-y[j-1])**2)/0.1)

kernel_size = 20
kernel = np.ones(kernel_size)/kernel_size
tvc = np.convolve(tv, kernel, mode='same')


for j in range(0,pv.size):
    k.append(j)


plt.figure(1, figsize=(6,4))
plt.subplot(2,1,1)
plt.plot(i, throttle, label = 'throttle')
plt.legend()
plt.subplot(2,1,2)
plt.plot(i, steering, label = 'steering')
plt.legend()
plt.savefig('image.png')

# plt.figure(1)
# plt.subplot(2,1,1)
# plt.plot(px, py, label = 'path')
# plt.plot(x,y, label = 'pos')
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(i,tvc,label='velocity')
# #plt.plot(k,pv,label='ref velocities')
# plt.ylim([-0.1,2])
# plt.legend()
# plt.savefig('image.png')