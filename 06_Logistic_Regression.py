'''To train a model to find out whether a given data is virginica or not.'''
from re import X
from sklearn import datasets,linear_model
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt 

iris= datasets.load_iris()
## print (iris.keys())
## print (iris.data)


'''Training a Logistic regression model for single instance. '''
x= iris.data[:,3:] # slicing the third index feature.
y = (iris.target==2).astype(np.int32) # Diplay binary output 1 or 0.'int32' or 'int64' can be used. 
clf= LogisticRegression()
clf.fit(x,y)
output_label = clf.predict([[4.5]])
# print (output_label)
'''Output:
Returns the binary output 0 or 1 whether a flower is Virginica for single instance. '''


'''Creating a new dataset to analyze values as input to the model for plotting.
Training a Logistic regression model for multiple instance'''
x_new = np.linspace(0,3,1000).reshape(-1,1) 
all_labels= clf.predict(x_new)
'''Output:
Returns the binary output 0 or 1 whether a flower is Virginica for 
the new 1000 instances (x_new) between values 0 and 3 we created above for testing.'''


'''Using Matplotlib to predict the probability of the 1000 dataset we created 
and plotting the same to check for the sigmoid function.'''
# y_prob= clf.predict_proba(x_new)
# # print (y_prob) # Prints 2 columns 
# # print (y_prob.shape)
# plt.plot (x_new,y_prob[:, 1]) # plotting x and y values. Also slicing the predicted 2 columns in 1 column.
# plt.show()








