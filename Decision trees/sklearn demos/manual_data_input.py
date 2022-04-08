# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
# import pickle

def custom_data(): 
    # set boundaries of plot
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
         
    # collect positive labels
    plt.title('Input planar dataset (right click for next labels)', fontweight ="bold")
    x1 = plt.ginput(n=-1, mouse_stop=3, mouse_pop=None)
    x1_plot = np.array(x1)
    plt.plot(x1_plot.T[0], x1_plot.T[1], 'o')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
    
    # collecte negative labels
    x2 = plt.ginput(n=-1, mouse_stop=3, mouse_pop=None)
    x2_plot = np.array(x2)
    plt.plot(x2_plot.T[0], x2_plot.T[1], 'x')
    plt.show()
    
    # collect and shuffle data
    dataset = []
    for x in x1:
        dataset.append((x, 0))
        
    for x in x2:
        dataset.append((x, 1))
        
    shuffle(dataset)
    
    # separate shuffled data in to data/targets
    data = []
    targets = []
    for x, y in dataset:
        data.append(x)
        targets.append(y)
    
    user_data = {'data': np.array(data),
                 'targets': np.array(targets)}
    
    return user_data
# with open("user_input_dataset.pickle", "wb") as f:
#     pickle.dump(user_data, f)
