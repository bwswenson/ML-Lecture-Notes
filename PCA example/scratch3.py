
import pickle

write = 0

if write:
    ## Save plot data if I like it
    my_data = (X,Y,stable_man,t0)
    
    # open a file, where you ant to store the data
    file = open('dim_3by10_sizept1bypt5_t0_20', 'wb')
    
    # dump information to that file
    pickle.dump(my_data, file)
    
    # close the file
    file.close()
else:
    ## Restore data
    # open a file, where you stored the pickled data
    file = open('dim_3by10_sizept1bypt5_t0_20', 'rb')
    
    # dump information to that file
    my_data = pickle.load(file)
    
    X_new = my_data[0]
    Y_new = my_data[1]
    stable_man_new = my_data[2]