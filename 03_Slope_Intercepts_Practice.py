import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes_x=np.array([[1],[2],[3],[4],[5]]) 

diabetes_x_train= diabetes_x
diabetes_x_test= diabetes_x 

'''Here label will lie on y axis'''
diabetes_y_train =np.array([[5],[7],[9],[11],[13]]) 
diabetes_y_test =np.array([[5],[7],[9],[11],[13]]) 


model = linear_model.LinearRegression() # specifying the type of anlysis to be performed. In this case linear regression
model.fit(diabetes_x_train,diabetes_y_train) # fitting and training data to plot x(features) and y(labels) axis.
diabetes_y_predict= model.predict(diabetes_x_test) # predicting labels based on the test features.

# print (diabetes_y_predict.dtype) # printing the predicted labels for test features.
'''Output :
[[2.5]
 [3. ]
 [3.5]]
float64 
'''
# print (diabetes_y_test.dtype) # printing the actual labels for the test features.
'''Output :
[[3]
 [2]
 [4]]
int32
'''
'''Why the o/p doesn't exactly match with the trained data for the same i/p?
It is because o/p is the best fit line that travels between the scattered points.
It ensures to minimise the error and doesn't copy the same values provided for training the labels even if they are same.
Remember : This is linear regression testing. So the line matters. '''



print ("Mean Squared Error is :", mean_squared_error(diabetes_y_test,diabetes_y_predict))
print ("Weights:", model.coef_)
print ("intercepts:", model.intercept_)

'''Output : Refer Notes for manual calculations using partial derivative.
Mean Squared Error is : 0.5000000000000001
Weights: [[0.5]]
intercepts: [2.]
'''

# plt.scatter(diabetes_x_train,diabetes_y_train)
plt.scatter(diabetes_x_test,diabetes_y_test) #For plotting features and labels. Note - If attribute error is shown import matplotlib.pyplot as plt
plt.plot(diabetes_x_test,diabetes_y_predict) #Displays fit line 
plt.show() # displays the plot

'''Conclusion: The best fit line is the optimal line for any model. It minimises the sum of squared error.
 If it is moved even a bit the sum of square error will increase for sure. '''

