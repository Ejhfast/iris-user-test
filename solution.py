import pickle

with open("iris-dict.pkl", "rb") as f:
    iris_data = pickle.load(f)

# iris_data.keys() => ['SepalLength', 'SepalWidth', 'PetalLength', 'Name', 'PetalWidth']

# class to name mapping: {'Iris-setosa': 0, 'Iris-virginica': 2, 'Iris-versicolor': 1}

# DOCS:
# Logistic Regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# Cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
# Numpy transpose: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.T.html

from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression()

dataX = np.array([iris_data['SepalLength'], iris_data['SepalWidth'], iris_data['PetalLength'], iris_data['PetalWidth']]).T
dataY = iris_data['Name']

model.fit(dataX, dataY)

from sklearn.model_selection import cross_val_score

print(cross_val_score(model, dataX, dataY, scoring="f1_macro", cv=15))
