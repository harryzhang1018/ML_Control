from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.signal

def dist(point, p2):
    return np.sqrt((point[0]-p2[0])**2+(point[1]-p2[1])**2)
t1 = np.loadtxt('./data/mc_bb_1.csv', delimiter=',')
t2 = np.loadtxt('./data/mc_bb_2.csv', delimiter=',')
t3 = np.loadtxt('./data/mc_bb_3.csv', delimiter=',')
t4 = np.loadtxt('./data/mc_bb_4.csv', delimiter=',')
t5 = np.loadtxt('./data/mc_bb_5.csv', delimiter=',')
traj = np.concatenate((t1, t2, t3, t4, t5), axis=0)
#path = genfromtxt('/home/scaldararu/Documents/Python_Data_Manipulation/test_vel/path.csv', delimiter=',')
path = np.loadtxt('./paths/lot17_sinsquare.csv', delimiter=',')
x = traj[:,0]
y = traj[:,1]
px = path[:,0]
py = path[:,1]
pv = path[:,3]
throttle = traj[:,6]
steering = traj[:,7]
error = []
i = []
k = []


for j in range(100,x.size):
    point = np.array([x[j], y[j]])
    p1 = np.array([px[0], py[0]])
    sd = dist(point, p1)
    sdi = 0
    for l in range(0,px.size):
        p1 = np.array([px[l], py[l]])
        if(dist(point, p1)<sd):
            sd = dist(point, p1)
            sdi = l
    error.append(sd)
    k.append(sdi)
    i.append(j)




plt.figure(1, figsize=(10,4))

plt.plot(k,error,label='error')
plt.xlabel("Reference State Index")
plt.ylabel("Absolute Error")
#plt.plot(k,pv,label='ref velocities')
#plt.ylim([-0.1,2])
plt.legend()
plt.savefig('image.png')