from keras_core.models import load_model
import tensorflow as tf
from numpy import genfromtxt
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def read_csv_into_np_array(file_path):
    # Load CSV data into a numpy array
    return np.loadtxt(file_path, delimiter=',')

def combine_csv_files(folder_path):
    # Initialize an empty list to store numpy arrays
    arrays_list = []

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through each file in the folder
    for file in file_list:
        # Check if the file is a CSV file
        if file.endswith('.csv'):
            # Read the CSV file into a numpy array
            file_path = os.path.join(folder_path, file)
            array = read_csv_into_np_array(file_path)
            arrays_list.append(array)

    # Concatenate all numpy arrays into a single big numpy array
    combined_matrix = np.concatenate(arrays_list)

    return combined_matrix

# Replace 'folder_path' with the path to the folder containing your CSV files
folder_path = './data/circle_Manual_0la_trainingData/'
train_set = combine_csv_files(folder_path)

error_state = train_set[:,2:6] #error states--inputs
ctrl_output = train_set[:,6:8] # control outputs
print(error_state.shape)
# Load the model from a file
model = load_model('keras_ml_learnMC_0la_merge.keras')

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
avg_err = [np.mean(abs(pre_steering-ctrl_output[:,1])),np.mean(abs(pre_throttle-ctrl_output[:,0]))]
print('Neural Network: Mean error for steering and throttle are: \n',avg_err)
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