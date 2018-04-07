# SVM is a binary classifier seperates into groups (+ and -)
# Best separating Hyperplane
# perpendicular bisectors to closest data points gives the greatest distance
# After getting a separating hyperplane u can deal with unknown points
# Intuition of SVM is done by finding best seperating hyperplane and then after we
# can easily classify new data points.
#yi refers to the class
# equation to derive support vector: yi(xi+w+b)-1 = 0


import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(2,-1)

prediction = clf.predict(example_measures)

print(prediction)