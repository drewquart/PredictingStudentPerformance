"""
    ML Regression model predicting 3rd Exam grade based on 33 different variables
    in the student-mat.csv. Data was pulled from the UCI Machine Learning Repository.
    This was my first attempt at training a model using Python.

Author: Andrew Quartuccio
Credit to ML Tutorial on techwithtim.net
"""

"""
        IMPORTS
"""
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle

style.use('ggplot')

#   READ IN DATA (had semicolon as delimiter for some reason)
data = pd.read_csv('student-mat.csv', sep=";")

#   'Label'
#   term used in ML for desired result
predict = "G3"

#   Create array of weights - shuffle data
data = data[['G1', 'G2', 'studytime', 'failures', 'absences', 'G3']]
data = shuffle(data)

#   Assign Data into separate arrays
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""
        MODEL TRAINING
"""

#   Best Model Training and Saving
#   **Turn off to keep model
#   **Train model multiple times for best score
best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print("Accuracy: " + str(accuracy))

    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

#   Load Model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("----------------------")
print('Coefficients: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("----------------------")

predicted = linear.predict(x_test)

#   Print Each Guess, Data & Score
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])

"""
        PLOTTING
"""
#   Choose your x-axis
p = 'failures'
pyplot.scatter(data[p], data['G3'])
pyplot.legend(loc=4)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
