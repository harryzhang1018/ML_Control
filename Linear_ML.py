from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

train_set_1 = genfromtxt('./data/mpc_sin_train1.csv', delimiter=',')
train_set_2 = genfromtxt('./data/mpc_sin_train2.csv', delimiter=',')
train_set_3 = genfromtxt('./data/mpc_sin_train3.csv', delimiter=',')
##combining training data into one big matrix
train_set = np.concatenate((train_set_1,train_set_2,train_set_3),axis=0)
#train_set = np.unique(train_set,axis=0)

error_state = train_set[:,6:10] #A
throttle = train_set[:,4] #b1
steering = train_set[:,5] #b2
# print(throttle.shape)
# print(steering.shape)
## applying linear method to the problem of Ax=b
# least square solver
throttle_control_ls = np.linalg.lstsq(error_state,throttle)[0] #x1
steer_control_ls = np.linalg.lstsq(error_state,steering)[0] #x2
print('LS:\nthrottle: ',throttle_control_ls)
print('steering: ',steer_control_ls)
pre_throttle = error_state@throttle_control_ls
pre_steering = error_state@steer_control_ls

avg_err = [np.mean(abs(pre_steering-steering)),np.mean(abs(pre_throttle-throttle))]
print('LS: Mean error for steering and throttle are: \n',avg_err)


# lasso solver

throttle_control_lasso = Lasso(alpha=0.00000001)
steer_control_lasso = Lasso(alpha=0.001)

throttle_control_lasso.fit(error_state, throttle)
steer_control_lasso.fit(error_state,steering)

print('Lasso:\nthrottle: ',throttle_control_lasso.coef_)
print('steering: ',steer_control_lasso.coef_)

pre_throttle_lasso = error_state@throttle_control_lasso.coef_
pre_steering_lasso = error_state@steer_control_lasso.coef_

avg_err_lasso = [np.mean(abs(pre_steering_lasso-steering)),np.mean(abs(pre_throttle_lasso-throttle))]
print('lasso: Mean error for steering and throttle are: \n',avg_err_lasso)

# visualize result

time_step = np.arange(0,pre_steering.shape[0])
plt.figure()
plt.subplot(2,2,1)
plt.plot(time_step,throttle,label='actual throttle')
plt.plot(time_step,pre_throttle,label='predict throttle by LS')
plt.legend()

plt.subplot(2,2,2)
plt.plot(time_step,steering,label='actual steering')
plt.plot(time_step,pre_steering,label='predict steering by LS')
plt.legend()

plt.subplot(2,2,3)
plt.plot(time_step,throttle,label='actual throttle')
plt.plot(time_step,pre_throttle_lasso,label='predict throttle by Lasso')
plt.legend()

plt.subplot(2,2,4)
plt.plot(time_step,steering,label='actual steering')
plt.plot(time_step,pre_steering_lasso,label='predict steering by Lasso')
plt.legend()
plt.show()