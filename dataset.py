import numpy as np
import random

def import_dataset(n_positive_data,n_negative_data,number_of_parameters,dataset_name):
    X = np.zeros((n_positive_data+n_negative_data,number_of_parameters))
    Y = np.zeros(n_positive_data+n_negative_data)
    Y[n_negative_data:] = np.ones(n_positive_data)
    
    for m in range(n_negative_data):
        f = open("dataset/"+dataset_name+"/negative/"+str(m+1)+".dat",'r')

        str_tmp=str.split(f.readline().strip())
        x = np.zeros(number_of_parameters)
        for p in range(number_of_parameters):
            x[p] = float(str_tmp[p])
        X[m] = x
        
    for m in range(n_positive_data):
        f = open("dataset/"+dataset_name+"/positive/"+str(m+1)+".dat",'r')

        str_tmp=str.split(f.readline().strip())
        x = np.zeros(number_of_parameters)
        for p in range(number_of_parameters):
            x[p] = float(str_tmp[p])
        X[m+n_negative_data] = x

    combined = list(zip(X, Y))
    random.shuffle(combined)
    
    X_shuffled = np.zeros((n_positive_data+n_negative_data,number_of_parameters))
    Y_shuffled = np.zeros(n_positive_data+n_negative_data)
    X_shuffled[:],Y_shuffled[:]=zip(*combined)
    
    X_shuffled = np.asarray(X_shuffled).astype(np.float32)
    Y_shuffled = np.asarray(Y_shuffled).astype(np.int32)
    return X_shuffled, Y_shuffled
