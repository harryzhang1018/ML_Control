from keras.models import load_model
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

train_set_1 = genfromtxt('./data/mpc_sin_train1.csv', delimiter=',')
train_set_2 = genfromtxt('./data/mpc_sin_train2.csv', delimiter=',')
train_set_3 = genfromtxt('./data/mpc_sin_train3.csv', delimiter=',')
##combining training data into one big matrix
train_set = np.concatenate((train_set_1,train_set_2,train_set_3),axis=0)
throttle = train_set[:,4] #b1
steering = train_set[:,5] #b2

error_state = train_set[:,6:10] # error states--inputs
ctrl_output = train_set[:,4:6] # control outputs
# Load the model from a file
model = load_model('keras_m1.h5')

# Use the model to make predictions
y_pred = model.predict(error_state)
pre_throttle = y_pred[:,0]
pre_steering = y_pred[:,1]
print(y_pred.shape)

time_step = np.arange(0,pre_steering.shape[0])
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time_step,throttle,label='actual throttle')
plt.plot(time_step,pre_throttle,label='predict throttle by NN')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time_step,steering,label='actual steering')
plt.plot(time_step,pre_steering,label='predict steering by NN')
plt.legend()

# plt.show()

## for training EKF

ekf_t1 = genfromtxt('./data/ekf_sin_t1.csv',delimiter=',')
ekf_t2 = genfromtxt('./data/ekf_sin_t2.csv',delimiter=',')
ekf_t3 = genfromtxt('./data/ekf_sin_t3.csv',delimiter=',')

# # ekf_train_set = np.concatenate( (ekf_t1, ekf_t2, ekf_t3),axis = 0 )
# train_inputs = ekf_t1[:, [0, 1, -2, -1]]
# train_inputs = train_inputs.reshape((-1,4))
ekf_model = load_model('keras_ekf_m1.h5')
nn_state_esti_t1 = ekf_model.predict(ekf_t1[:, [0, 1, -2, -1]],verbose=0)
nn_state_esti_t2 = ekf_model.predict(ekf_t2[:, [0, 1, -2, -1]],verbose=0)
nn_state_esti_t3 = ekf_model.predict(ekf_t3[:, [0, 1, -2, -1]],verbose=0)
plt.figure(figsize=(10,5))
plt.subplot(3,1,1)
plt.plot(ekf_t1[:,0],ekf_t1[:,1],label='GPS measurement')
plt.plot(ekf_t1[:,2],ekf_t1[:,3],label='Ground Truth')
plt.plot(ekf_t1[:,4],ekf_t1[:,5],label='EKF')
plt.plot(nn_state_esti_t1[:,0],nn_state_esti_t1[:,1],label='NN State Estimation')
plt.legend()

plt.subplot(3,1,2)
plt.plot(ekf_t2[:,0],ekf_t2[:,1],label='GPS measurement')
plt.plot(ekf_t2[:,2],ekf_t2[:,3],label='Ground Truth')
plt.plot(ekf_t2[:,4],ekf_t2[:,5],label='EKF')
plt.plot(nn_state_esti_t2[:,0],nn_state_esti_t2[:,1],label='NN State Estimation')
plt.legend()

plt.subplot(3,1,3)
plt.plot(ekf_t3[:,0],ekf_t3[:,1],label='GPS measurement')
plt.plot(ekf_t3[:,2],ekf_t3[:,3],label='Ground Truth')
plt.plot(ekf_t3[:,4],ekf_t3[:,5],label='EKF')
plt.plot(nn_state_esti_t3[:,0],nn_state_esti_t3[:,1],label='NN State Estimation')
plt.legend()

plt.show()