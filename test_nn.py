from keras.models import load_model
from numpy import genfromtxt
import numpy as np
import time
import matplotlib.pyplot as plt

train_set_1 = genfromtxt('./data/sin_sim_testing_1.csv', delimiter=',')
train_set_2 = genfromtxt('./data/sin_sim_testing_2.csv', delimiter=',')
train_set_3 = genfromtxt('./data/sin_sim_testing_3.csv', delimiter=',')
train_set_4 = genfromtxt('./data/sin_sim_testing_4.csv', delimiter=',')
train_set_5 = genfromtxt('./data/sin_sim_testing_5.csv', delimiter=',')

##combining training data into one big matrix
train_set = np.concatenate((train_set_1,train_set_2,train_set_3,train_set_4,train_set_5),axis=0)
error_state = train_set[:,2:6] #error states--inputs
ctrl_output = train_set[:,6:8] # control outputs
# Load the model from a file
model = load_model('keras_ml_learnMC.h5')

# Use the model to make predictions
print(error_state.shape)
y_pred = model.predict(error_state)
pre_throttle = y_pred[:,0]
pre_steering = y_pred[:,1]
print(y_pred.shape)

err = np.array([0,0,0,0])
error_state_test = err.reshape(1,-1)
print(error_state_test.shape)
t_start = time.time()
ctrl = model.predict(error_state_test)
# throttle = ctrl[0,0]
# steering = ctrl[0,1]
timespan = time.time()-t_start
print('time takes: ',timespan)
print('shape of output: ', ctrl.shape)
time_step = np.arange(0,pre_steering.shape[0])
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time_step,ctrl_output[:,0],label='actual throttle')
plt.plot(time_step,pre_throttle,label='predict throttle by NN')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time_step,ctrl_output[:,1],label='actual steering')
plt.plot(time_step,pre_steering,label='predict steering by NN')
plt.legend()

plt.show()