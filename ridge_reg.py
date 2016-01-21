import pandas as pd
import numpy as np


def get_numpy_data(df, features, output):
    '''
    The function takes a dataset, a list of features (e.g. [‘sqft_living’, ‘bedrooms’]) to be used as inputs, 
    and a name of the output (e.g. ‘price’). It returns a ‘features_matrix’ (2D array) consisting of 
    a column of ones followed by columns containing the values of the input features in the data set 
    in the same order as the input list. It also returns an ‘output_array’, 
    which is an array of the values of the output in the dataset (e.g. ‘price’).
    '''
    n=len(df)
    ff=pd.DataFrame()
    ff['constant'] = np.ones(n) # add a constant column to an SFrame  

    for f in features:
        ff[f]=df[f]    
    #print type(ff)
    
    features_matrix = ff.values    
    #print type(features_matrix)
   
    output_sarray = df[output]    
    output_array = output_sarray.values
    
    return(features_matrix, output_array)

def ridge_regression(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    
    weights = np.array(initial_weights) # make sure it's a numpy array
    count=0
    
    while count <= max_iterations:
        preds=np.dot(feature_matrix,weights)     
        errors=preds-output 
        for i in xrange(len(weights)):          
            if (i==0):
                derivative=2*np.dot(feature_matrix[:, i],errors)
            else:
                derivative=2*np.dot(feature_matrix[:, i],errors) + 2*l2_penalty*weights[i]
            weights[i]=weights[i]-step_size*derivative
        count = count + 1
    return weights


def predict_output(feature_matrix, weights):
    predictions=np.dot(feature_matrix,weights)
    return(predictions)