from pyexpat import features
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()

# print (iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
# print (iris.DESCR)
'''Output : 
 :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa- 0
                - Iris-Versicolour- 1
                - Iris-Virginica- 2'''

features= iris.data
labels= iris.target
# print (features[0],labels[0])
# print (features[:-50])

clf= KNeighborsClassifier()
clf.fit (features,labels)
pred = clf.predict([[6,3,4,2]])
print (pred)
