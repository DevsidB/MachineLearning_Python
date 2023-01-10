# refer jupyter notebook 'python in one shot' for details regarding slicing an array. 
# refer jupyter 'machine learning' notebook for experiment :)

# import matplotlib as plt #scatter, show methods will show attribute error for this import. Use below line.
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

'''To access the list of datasets contained inside datasets 
type --> 'datsets.load_' and then press tab to access all the preloaded datasets inside. '''

diabetes = datasets.load_diabetes()

'''These are some of the basic commands to understand in depth about our data.'''

# print (diabetes.keys()) 
#'data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'
# print (diabetes.data)
# print (diabetes.DESCR) 
# print (diabetes.target) # Another attribute named target has a data with one column and 442 rows check DESCR
# print (diabetes.target.shape)

diabetes_x= diabetes.data[:,np.newaxis,9] # Keep changing the features index to see how slope changes with each changing feature but intercept remains same.
# extracting 2nd index feature and arranging it in form of a row in single column.
# print(diabetes_x)
# print (diabetes_x.shape)

'''All the features are to be plotted on x axis. 
In this case we will  only take 1 feature so that the line of fit can be plotted.
If multiple features are taken into consideration line of best fit becomes a plane with multiple lines 
Accuracy is more for more features '''

diabetes_x_train= diabetes_x[:-30] # slicing the feature data for training - (excluding last 30 )
# print (diabetes_x_train[:2])
# print (diabetes_x_train)

diabetes_x_test= diabetes_x[-30:] # slicing the feature data for testing  - (only last 30)
# print (diabetes_x_test.reshape(1,30))
'''Output: For manual calculation using formula. y = mx + b (substitute x from here)
[[ 0.08540807 -0.00081689  0.00672779  0.00888341  0.08001901  0.07139652
  -0.02452876 -0.0547075  -0.03638469  0.0164281   0.07786339 -0.03961813
   0.01103904 -0.04069594 -0.03422907  0.00564998  0.08864151 -0.03315126
  -0.05686312 -0.03099563  0.05522933 -0.06009656  0.00133873 -0.02345095
  -0.07410811  0.01966154 -0.01590626 -0.01590626  0.03906215 -0.0730303 ]]
'''

'''Here target will be our label which will lie on y axis'''
diabetes_y_train =diabetes.target[:-30] # slicing the label data for training - (excluding last 30 )
# print (diabetes_y_train.shape)

diabetes_y_test =diabetes.target[-30:]  # slicing the label data for testing  - (only last 30)
# print (diabetes_y_test.shape)

model = linear_model.LinearRegression() # specifying the type of anlysis to be performed. In this case linear regression
model.fit(diabetes_x_train,diabetes_y_train) # fitting and training data to plot x(features) and y(labels) axis.
diabetes_y_predict= model.predict(diabetes_x_test) # predicting labels based on the test features.

# print (diabetes_y_predict) # printing the predicted labels for test features.
'''Output : y = mx + b (substitute y from here)
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

'''Output : When 2nd index feature is considered.
Mean Squared Error is : 3035.060115291269

Weights: [941.43097333] 
How ? 
y = mx + b 
y = 233.80294072 (first predicted value )
x = 0.08540807 (first test value)
b = 153.39713623331644 (from below)
m = (233.80294072 - 153.39713623331644)/ 0.08540807
m = 941.43097333 (cross verified manually)
Same can be done for finding weights(slope) for each further instance. SAME RESULTS

m = (152.62808714 - 153.39713623331644) / -0.00081689
m = 941.43097333 and so on ...

# Only 1 weight returned as number of feature is 1. More the feaures more the weights.
# Weight is the slope of the line for a given feature. Here only one feature 1 slope.
# Any further points of y corresponding to y axis can be plotted on this line for any values of x on x axis given the slope. 
# tan theta = opp side/Adjside or  m= (y-y1)/(x-x1) or m = rise /run

intercepts: 153.39713623331644'''

# plt.scatter(diabetes_x_train,diabetes_y_train)
plt.scatter(diabetes_x_test,diabetes_y_test) #For plotting features and labels. Note - If attribute error is shown import matplotlib.pyplot as plt
plt.scatter(diabetes_x_test,diabetes_y_predict) # All predicted points on line
plt.plot(diabetes_x_test,diabetes_y_predict) #Displays fit line 
plt.show() # displays the plot

'''
0      - age     age in years
1      - sex
2      - bmi     body mass index
3      - bp      average blood pressure
4      - s1      tc, total serum cholesterol
5      - s2      ldl, low-density lipoproteins
6      - s3      hdl, high-density lipoproteins
7      - s4      tch, total cholesterol / HDL
8      - s5      ltg, possibly log of serum triglycerides level
9      - s6      glu, blood sugar level
'''
'''When : diabetes_x= diabetes.data[:,np.newaxis,0]  
Keep changing value of 0''' 

'''Output : When 0th index feature is considered.age
Mean Squared Error is : 5275.139226727016
Weights: [298.46194553]
intercepts: 153.4350067260719
'''
'''Output : When 1st index feature is considered.sex
Mean Squared Error is : 5497.755867852905
Weights: [54.6011682]
intercepts: 153.47896015821865
'''
'''Output : When 2nd index feature is considered.bmi
Mean Squared Error is : 3035.060115291269
Weights: [941.43097333]
intercepts: 153.39713623331644
'''
'''Output : When 3rd index feature is considered.bp
Mean Squared Error is : 3428.9992657001594
Weights: [693.36333298]
intercepts: 153.19151004427027
'''
'''Output : When 4th index feature is considered.s1 
Mean Squared Error is : 5652.432806871776
Weights: [360.59688657]
intercepts: 153.8911783292527
'''
'''Output : When 5th index feature is considered.s2
Mean Squared Error is : 5594.520132369854
Weights: [294.72241201]
intercepts: 153.89533162927725
'''
'''Output : When 6th index feature is considered.s3 observe the negative slope here.
Mean Squared Error is : 4594.8246219761395
Weights: [-645.5284632]
intercepts: 153.2144350252234
'''
'''Output : When 7th index feature is considered.s4
Mean Squared Error is : 4934.778556976869
Weights: [706.19817526]
intercepts: 153.71943629239087
'''
'''Output : When 8th index feature is considered.s5
Mean Squared Error is : 3312.2220523707397
Weights: [900.20186421]
intercepts: 153.10182488328562
'''
'''Output : When 9th index feature is considered.s6
Mean Squared Error is : 5040.185152996671
Weights: [626.5337537]
intercepts: 154.16117136182055
'''