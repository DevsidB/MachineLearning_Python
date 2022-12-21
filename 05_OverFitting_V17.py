'''Aim: 
1. To use the training data points for the testing purpose.
2. Overfitting is the process in which we get high accuracy when 'training data' itself is used for the testing purpose.Output is not approximated but exact for the data shown to the model.
3. High accuracy for 'training data' used as the 'test data' means that for the same values provided we will 
get accurate and most precise data. But once the data is channged and model is provided with data other than the 
one previously known by it, model will distort the output severely and extreme error will occur for the predicted data.
4. So it is necessary to ensure that the data is not overfitted.

Here we provide the 'training features' and 'labels' to the model and then asks the model to generate the 'labels' 
for the same 'training data' we provided as the 'test data'.
If model shows very minimum error we need to rethink ! We will get exact data for the values shown to model
but when random data is thrown the model is likely to generate an extremely large error.
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
diabetes_x= diabetes.data[:,np.newaxis,2] # extracting 2nd index feature and arranging it in form of a row in single column.

diabetes_x_train= diabetes_x[:-30] # slicing the feature data for training - (excluding last 30 )
diabetes_x_test= diabetes_x[-30:] # slicing the feature data for testing  - (only last 30)
print (diabetes_x_test.shape)

diabetes_y_train =diabetes.target[:-30] # slicing the label data for training - (excluding last 30 )
diabetes_y_test =diabetes.target[-30:]  # slicing the label data for testing  - (only last 30)

model = linear_model.LinearRegression() # specifying the type of anlysis to be performed. In this case linear regression
model.fit(diabetes_x_train,diabetes_y_train) # fitting and training data to plot x(features) and y(labels) axis.
diabetes_y_predict= model.predict(diabetes_x_train) # Asking model to predict labels input -> training features.

# print (diabetes_y_predict) # printing the predicted labels for test features.
# print (diabetes_y_train) # printing the actual labels for the test features.
# print (diabetes_y_test) # printing the actual labels for the test features.

print ("Mean Squared Error is :", mean_squared_error(diabetes_y_train,diabetes_y_predict)) # Difference between actual values and predicted values.
print ("Weights:", model.coef_)
print ("intercepts:", model.intercept_)

plt.scatter(diabetes_x_train,diabetes_y_train) # Plotting Original points provided during training.
plt.scatter(diabetes_x_train,diabetes_y_predict) # Plotting points predicted by model.
plt.plot(diabetes_x_train,diabetes_y_predict) #Displays fit line 
plt.show() # displays the plot


# Conlusion:
'''We trained the data and tested for the same training data. 
SSE is not zero,or even close to it. 
SSE incrased a little more than Single feature code
Hence we can say that out model is not overfitted.'''

'''Output (Current Code using 'training data' as 'test data'): 
Mean Squared Error is : 3954.611332145007 # The SSE is bit more than Single feature code.
Weights: [941.43097333]
intercepts: 153.39713623331644
'''
'''Output (Diabetes Single feature): 
Mean Squared Error is : 3035.060115291269
Weights: [941.43097333] 
# Only 1 weight returned as number of feature is 1. More the feaures more the weights.
# Weight is the slope of the line.Here only one feature 1 slope.
# tan theta = opp side/Adj side
intercepts: 153.39713623331644'''