# refer jupyter notebook 'python in one shot' for details regarding slicing an array. 
# refer jupyter 'machine learning' notebook for experiment :)

# import matplotlib as plt #scatter, show methods will show attribute error for this import. Use below line.
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

'''To access the list of datasets contained inside datasets 
type --> 'datsets.' and then press tab to access all the preloaded datasets inside. '''

diabetes = datasets.load_diabetes()

'''These are some of the basic commands to understand in depth about our data.'''

# print (diabetes.keys()) 
#'data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'
# print (diabetes.data)
# print (diabetes.DESCR) 
# print (diabetes.target) # Another attribute named target has a data with one column and 442 rows check DESCR
# print (diabetes.target.shape)

diabetes_x= diabetes.data # extracting 2nd index feature and arranging it in form of a row in single column.
# print(diabetes_x)
print (diabetes_x.shape)

'''All the features are to be plotted on x axis. 
In this case we will  only take 1 feature so that the line of fit can be plotted.
If multiple features are taken into consideration line of best fit becomes a plane with multiple lines 
Accuracy is more for more features '''

diabetes_x_train= diabetes_x[:-30] # slicing the feature data for training - (excluding last 30 )
# print (diabetes_x_train[:2])
print (diabetes_x_train.shape)

diabetes_x_test= diabetes_x[-30:] # slicing the feature data for testing  - (only last 30)
print (diabetes_x_test.shape)

'''Here target will be our label which will lie on y axis'''
diabetes_y_train =diabetes.target[:-30] # slicing the label data for training - (excluding last 30 )
print (diabetes_y_train.shape)

diabetes_y_test =diabetes.target[-30:]  # slicing the label data for testing  - (only last 30)
print (diabetes_y_test.shape)

model = linear_model.LinearRegression() # specifying the type of anlysis to be performed. In this case linear regression
model.fit(diabetes_x_train,diabetes_y_train) # fitting and training data to plot x(features) and y(labels) axis.
diabetes_y_predict= model.predict(diabetes_x_test) # predicting labels based on the test features.

# print (diabetes_y_predict) # printing the predicted labels for test features.
'''Output : 
[233.80294072 152.62808714 159.73088683 161.76025817 228.72951237
 220.61202701 130.3050024  101.89380365 119.14346004 168.86305786
 226.70014103 116.09940303 163.78962951 115.08471736 121.17283138
 158.71620116 236.84699773 122.18751705  99.86443231 124.21688839
 205.39174197  96.8203753  154.65745848 131.31968807  83.62946159
 171.90711487 138.42248776 138.42248776 190.17145692  84.64414726]'''
# print (diabetes_y_test) # printing the actual labels for the test features.
'''Output :
[261. 113. 131. 174. 257.  55.  84.  42. 146. 212. 233.  91. 111. 152.
 120.  67. 310.  94. 183.  66. 173.  72.  49.  64.  48. 178. 104. 132.
 220.  57.]'''


'''As the values are predicted. We now need to compare the same with our actual test values.'''
'''Mean square error will allow us to find the difference between preicted and actual values'''
'''Mean square error = [(predicted1 - actual1)^2+(predicted2 - actual2)^2+...(predictedn - actualn)^2] divided by n '''
'''Here we are finding the average of all the error values '''
'''In short - Mean square error= average of sum of squared error'''

print ("Mean Squared Error is :", mean_squared_error(diabetes_y_test,diabetes_y_predict))
print ("Weights:", model.coef_)
print ("intercepts:", model.intercept_)

'''Output :
Mean Squared Error is : 1826.484171279504
Weights: [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
  458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ] 
# Here total 10 weights are displayed and error is also minimised as more features are present.
# Each weight is the slope for that individual feature. 
intercepts: 153.05824267739402'''

# Scatter Plot will not work for multiple weights. Best fit line will be displayed only for single feature. For all features it will become a plane. 
# plt.scatter(diabetes_x_train,diabetes_y_train)
# plt.scatter(diabetes_x_test,diabetes_y_test) #For plotting features and labels. Note - If attribute error is shown import matplotlib.pyplot as plt
# plt.plot(diabetes_x_test,diabetes_y_predict) #Displays fit line 
# plt.show() # displays the plot

'''Conclusion : 
Here total 10 weights are displayed and error is 
also minimised as more features are present which improves accuracy .'''

'''Output of single feature: 
Mean Squared Error is : 3035.060115291269
Weights: [941.43097333] # Only 1 weight returned as number of feature is 1. More the feaures more the weights.
intercepts: 153.39713623331644'''

'''Output considering all features :
Mean Squared Error is : 1826.484171279504
Weights: [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
  458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ] # Here total 10 weights are displayed and error is also minimised as more features are present.
intercepts: 153.05824267739402'''
