# Train a logistic regression classifier to predict whether a flower is iris virginica or not
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# X = features, we are taking all the features
# based on the sepal length,sepal width, peta length, petal width
# we will be classifying if a flower is Iris- Virginica or not
X = iris["data"]

# Y = labels
Y = (iris["target"] == 2).astype(np.int64)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X,Y)

example1 = clf.predict(([[0.1,3,0.3,0.6]]))
print(example1)

example2 = clf.predict(([[5,2.3,19,2.6]]))
print(example2)

# creating 4 columns in each row- 1 column each for sepal length, sepal width, petal length and petal width
X_new = np.linspace(0,5,10000,endpoint = False).reshape(-1,4)
# print(X_new)

# plotting the probability that a flower
# with given sepal length,sepal width, petal length and petal width is iris-virginica or not
y_prob = clf.predict_proba(X_new)
print(y_prob)

plt.plot(X_new, y_prob[:,1], "g-")
plt.show()