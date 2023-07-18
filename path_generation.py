import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# filename = './data/sin_test_v1.csv'
filename = './data/Circle_Traj_CW.csv'

def convert_conefile(file,move_to_dART):
    data = genfromtxt(file,delimiter=',')
    x = data[:,0]
    y = data[:,1]
    print(x.shape)
    seq = np.arange(x.shape[0])[:, np.newaxis]
    print(seq.shape)
    color_code = np.ones((x.shape[0],1))
    cone_file = np.hstack((seq, color_code, x[:, np.newaxis], y[:, np.newaxis]))
    
    np.savetxt('circle.csv', cone_file, delimiter=',',fmt='%f')
    if move_to_dART:
        np.savetxt('/home/harry/autonomy-research-testbed/sim/data/autonomy-toolkit/paths/circle.csv', cone_file, delimiter=',',fmt='%f')

convert_conefile(filename,True)