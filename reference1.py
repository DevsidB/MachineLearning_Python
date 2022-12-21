import numpy as np
x= np.array([[1,2,3,4,5,6,7,8],[11,12,13,14,15,16,17,18],[21,22,23,24,25,26,27,28]])
# x= np.array([1,2,3,4,5,6,7,8])
# array1=np.array([[2,0,4,5,6],[4,5,6,0,12]])
# listarray = np.array([[1,2,3], [4,5,6], [7,8,9]])

# print (x.shape)
# print (x.size)
# print(x[0])
# print (x)
# print (array1.shape)
# print (listarray.shape)

# x= x[np.newaxis,:]
# print (x)

# x= x[:,np.newaxis]
# print (x)

# x= x[np.newaxis,2:7]
# print (x)

# x= x[1:3,np.newaxis,1] 
# ''' 1:3 denotes from array 1 to 2, 
# np.newaxis at 1st index in code arranges vertically in rows and adds a dimension,
# 1 denotes the column number when arrays are arranged as matrix.'''
# print (x)

# x= x[np.newaxis,1:3,1] 
# ''' 1:3 denotes from array 1 to 2, 
# np.newaxis at 0th index in code arranges horizontally in columns and adds a dimension,
# 1 denotes the column number from array when arrays are arranged as matrix.'''
# print (x)

# x= x[1:3,np.newaxis,1:7] 
# ''' 1:3 denotes from array 1 to 2, 
# np.newaxis at 1st index in code arranges vertically in rows and adds a dimension,
# 1:7 denotes the column number when arrays are arranged as matrix.'''
# print (x)

x= x[np.newaxis,1:3,1:7] 
''' 1:3 denotes from array 1 to 2, 
np.newaxis at 0th index in code arranges horizontally in columns and adds a dimension,
1 denotes the column number from array when arrays are arranged as matrix.'''
print (x)