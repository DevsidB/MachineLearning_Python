import numpy as np
x= np.array([1,2,3,4,5,6,7,8])
print (x[0])
x= x[-6:] # used for testing as it only considers the last 6
print (x)

x= np.array([1,2,3,4,5,6,7,8])
x= x[:-6] # Can be used for training Large sample as it doesnt consider the last 6
print (x)
