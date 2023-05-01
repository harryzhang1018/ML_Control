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
plt.figure()
plt.subplot(2,1,1)
plt.plot(time_step,throttle,label='actual throttle')
plt.plot(time_step,pre_throttle,label='predict throttle by NN')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time_step,steering,label='actual steering')
plt.plot(time_step,pre_steering,label='predict steering by NN')
plt.legend()

plt.show()