# an example of linear regression by more comprehensive method

#Step 1: importing necessary python package

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.metrics as ms
import matplotlib.pyplot as plt

#Step2: Importing dataset

data = pd.read_csv ('/home/kishor/PycharmProjects/my_hello_world/Engvsfuel.csv')
X = data.iloc [:, 0].values.reshape(-1, 1) # values converts it into a numpy array with value normalized to range of -1 to 1
Y = data.iloc [:, 1].values.reshape(-1, 1) # data.iloc[:, 1] will keep Y as second column in the python array while number of row being unknown and left it to be determined by the python itself.
linear_regression = LinearRegression() # create object for the class
linear_regression.fit(X, Y) #perform linear regression
Y_pred = linear_regression.predict(X) # prediction of the value of y based on the value of X

#Step3: Visualizing the linear regression

plt.scatter(X, Y, c = 'g')
plt.plot(X, Y_pred, c = 'b')
plt.xlabel('Engine Size')
plt.ylabel('Fuel consumption in highway')
plt.title('Engine size vs fuel consumption in the highway')
plt.show()


