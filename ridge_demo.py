import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  ridge_reg import ridge_regression , get_numpy_data , predict_output


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 
'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str,
'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


train_data=pd.read_csv("kc_house_train_data.csv",dtype=dtype_dict)
test_data=pd.read_csv("kc_house_test_data.csv",dtype=dtype_dict)

print '\n-------------------------------\n'
print 'Ridge regression using using sqft_living as the feature \n'
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000
l2_penalty=0.0
simple_weights_0_penalty=ridge_regression(simple_feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
print "The coefficient of sqft_living with 0 penalty is %.1f" %(simple_weights_0_penalty[1])

l2_penalty=1e11
simple_weights_high_penalty=ridge_regression(simple_feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
print "The coefficient of sqft_living with high penalty is %.1f" %(simple_weights_high_penalty[1])

test_preds=predict_output(simple_test_feature_matrix,simple_weights_0_penalty)
RSS=np.sum((test_preds-test_output)**2)
print '\nRSS on test data with 0 penalty',RSS

test_preds=predict_output(simple_test_feature_matrix,simple_weights_high_penalty)
RSS=np.sum((test_preds-test_output)**2)
print 'RSS on test data with high penalty',RSS

test_preds=predict_output(simple_test_feature_matrix,initial_weights)
RSS=np.sum((test_preds-test_output)**2)
print 'RSS on test data with zero initial_weights',RSS




print '\n-------------------------------\n'
print 'Ridge regression using using sqft_living and sqft_living15 as features\n'
# sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
model_features = ['sqft_living', 'sqft_living15'] 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000
l2_penalty=0.0
multi_weights_0_penalty=ridge_regression(feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
print "The coefficient of sqft_living with 0 penalty is %.1f" %(multi_weights_0_penalty[1])

l2_penalty=1e11
multi_weights_high_penalty=ridge_regression(feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
print "The coefficient of sqft_living with high penalty is %.1f" %(multi_weights_high_penalty[1])

test_preds=predict_output(test_feature_matrix,multi_weights_0_penalty)
RSS=np.sum((test_preds-test_output)**2)
print '\nRSS on test data with 0 penalty',RSS

test_preds=predict_output(test_feature_matrix,multi_weights_high_penalty)
RSS=np.sum((test_preds-test_output)**2)
print 'RSS on test data with high penalty',RSS

test_preds=predict_output(test_feature_matrix,initial_weights)
RSS=np.sum((test_preds-test_output)**2)
print 'RSS on test data with zero initial_weights',RSS