# imporiting the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# getting address of the file
path = input ('Enter the csv file full address')

print ('\n \n entered path is: \n \n ', path)

df = pd.read_csv(path)

#print (df)

#visualizing data more meaningfully

#from seaborn
#import seaborn as sns
#fig = sns.pairplot(df)

# getting statistical meaining of dataset

print (df.describe())

#checking for null value

df.isnull().sum()

X = df.iloc[:, :-1]
y = df.iloc[:, -1] # assuming only one outcome value

print ('\n \n Features: \n', X)

print ('\n \n Target Variable:\n', y)


#splitting training and testing data
from sklearn.model_selection import train_test_split

size = df.shape

print(f'''\n \n \n number of total rows: {size[0]} Is it exactly 20% computable? If not please reconsider the test_size in case if error occurs.''') #for number of total number of rows

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size =0.20, random_state = 0);


task = input ('Select options for your desired operations: \n 1 => logistic regression \n 2 => linear reggression \n \n')

print ('selected task is: ', task)

task = str (task)

# chossing the model
if task=='1':
    # model development and prediction

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()

    # fitting th model with data

    lr.fit(X_train, y_train);

    # prediction of value
    y_pred = lr.predict(X_test)

    print('Predicted values: \n', y_pred)
    print('\n testing data', y_test)

    # model evaluation using confusion matrix

    from sklearn.metrics import confusion_matrix

    confmat = confusion_matrix(y_test, y_pred)
    from sklearn import metrics
    print('\n Confusion matrix is : \n', confmat)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('Precision: ', metrics.precision_score(y_test, y_pred))
    print('Recall: ', metrics.recall_score(y_test, y_pred))


elif task=='2': # for linear regression model
    # model development and prediction

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()

    # fitting th model with data

    lr.fit(X_train, y_train)

    # prediction of value
    y_pred = lr.predict(X_test)

    print('Predicted values: \n', y_pred)
    print('\n testing data', y_test)

    # model evaluation using confusion matrix

    from sklearn.metrics import confusion_matrix

    confmat = confusion_matrix(y_test, y_pred)
    from sklearn import metrics

    print('\n Confusion matrix is : \n', confmat)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('Precision: ', metrics.precision_score(y_test, y_pred))
    print('Recall: ', metrics.recall_score(y_test, y_pred))



else:
    print("Error in task selection. Please do it again properly!")

