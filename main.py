import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')

print('-'*25)
print('EDA')
print('-'*25)
print('Shape of Dataset: ', dataset.shape)
print('-'*25)
print('Describe: \n', dataset.describe())

print('\n'*2)
print('-'*25)
print('Preparing data')
print('-'*25)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
print('X: ', X)
print('-'*25)
print('Y: ', y)
print('-'*25)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('\n'*2)
print('-'*25)
print('Training the Algorithm......')
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# print('Intercept = ')
# print(regressor.intercept_)

# print('Coefficient of X = ')
# print(regressor.coef_)

print('-'*25)
print('Making Predictions')
print('-'*25)

y_pred = regressor.predict(X_test)
print('Y test: \n', y_test)
print('-'*25)
print('Predicted Y: \n', y_pred)
print('-'*25)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
print('-'*25)


print('Evaluating the Algorithm')

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('-'*25)


print('You can see that the value of root mean squared error is 4.64, which is less than 10% of the mean value of the percentages of all the students i.e. 51.48. This means that our algorithm did a decent job.')
print('-'*25)

print('Regressor Score: ', regressor.score(X_train, y_train))
print('-'*25)