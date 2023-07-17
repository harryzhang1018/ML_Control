from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

train_set_1 = genfromtxt('./data/sin_sim_testing_1.csv', delimiter=',')
train_set_2 = genfromtxt('./data/sin_sim_testing_2.csv', delimiter=',')
train_set_3 = genfromtxt('./data/sin_sim_testing_3.csv', delimiter=',')
train_set_4 = genfromtxt('./data/sin_sim_testing_4.csv', delimiter=',')
train_set_5 = genfromtxt('./data/sin_sim_testing_5.csv', delimiter=',')

##combining training data into one big matrix
train_set = np.concatenate((train_set_1,train_set_2,train_set_3,train_set_4,train_set_5),axis=0)
error_state = train_set[:,2:6] #A

throttle = train_set[:,6] #b1
print(throttle)
steering = train_set[:,7] #b2

# least square solver
throttle_control_ls = np.linalg.lstsq(error_state,throttle)[0] #x1
steer_control_ls = np.linalg.lstsq(error_state,steering)[0] #x2
print('LS:\nthrottle: ',throttle_control_ls)
print('steering: ',steer_control_ls)
pre_throttle = error_state@throttle_control_ls
pre_steering = error_state@steer_control_ls

avg_err = [np.mean(abs(pre_steering-steering)),np.mean(abs(pre_throttle-throttle))]
print('LS: Mean error for steering and throttle are: \n',avg_err)

# visualize result

time_step = np.arange(0,pre_steering.shape[0])
plt.figure()
plt.subplot(2,1,1)
plt.plot(time_step,throttle,label='actual throttle')
plt.plot(time_step,pre_throttle,label='predict throttle by LS')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time_step,steering,label='actual steering')
plt.plot(time_step,pre_steering,label='predict steering by LS')
plt.legend()

plt.show()