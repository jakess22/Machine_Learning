import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report


# Correlation Matrix
df = pd.read_csv("auto-mpg.csv")
#del df['origin']
#d.head()
sns.heatmap(df.corr(), annot = True, square = True, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm') 

# Pair-Plot
sns.pairplot(df, hue = "origin", diag_kind = "origin")

## Linear and Polynomial Regression

# Split dataset
X = df['weight'].values
y = df['mpg'].values

X = X[:, np.newaxis]
y = y[:, np.newaxis]

X_train = X[:-20]
y_train = y[:-20]

X_test = X[-20:]
y_test = y[-20:]

# Simple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train) # mpg train prediction
y_test_pred = model.predict(X_test)   # mpg test prediction

mse = model.score(X, y)
print(f"MSE = {mse}")

plt.scatter(X_train, y_train, color = "black", s = 10)
plt.plot(X_train, y_train_pred, color = "green", linewidth = 3)
plt.show()

# Polynomial Regression Degree 2
X = df['weight'].values
y = df['mpg'].values

X = X[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree = 2)
X_poly = polynomial_features.fit_transform(X)

X_poly_train = X_poly[:-20]
y_train = y[:-20]

X_poly_test = X_poly[-20:]
y_test = y[-20:]

model = LinearRegression()
model.fit(X_poly_train, y_train)
y_train_pred = model.predict(X_poly_train)
y_test_pred = model.predict(X_poly_test)

rmse_test = np.sqrt(mean_squared_error(y_test,y_test_pred))
print('\nTest RMSE: %8.15f' % rmse_test)

rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
print('\nTraining RMSE: %8.15f' % rmse)


plt.scatter(X, y, s=10)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_train_pred), key=sort_axis)
X_poly_train, y_train_pred = zip(*sorted_zip)
plt.plot(X_poly_train, y_train_pred, color='m')
plt.show()

# Polynomial Regression Degree 3

X = df['weight'].values
y = df['mpg'].values

X = X[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree = 3)
X_poly = polynomial_features.fit_transform(X)

X_poly_train = X_poly[:-20]
y_train = y[:-20]

X_poly_test = X_poly[-20:]
y_test = y[-20:]

model = LinearRegression()
model.fit(X_poly_train, y_train)
y_train_pred = model.predict(X_poly_train)
y_test_pred = model.predict(X_poly_test)

rmse_test = np.sqrt(mean_squared_error(y_test,y_test_pred))
print('\nTest RMSE: %8.15f' % rmse_test)

rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
print('\nTraining RMSE: %8.15f' % rmse)


plt.scatter(X, y, s=10)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_train_pred), key=sort_axis)
X_poly_train, y_train_pred = zip(*sorted_zip)
plt.plot(X_poly_train, y_train_pred, color='m')
plt.show()

# Polynomial Regression Degree 4

X = df['weight'].values
y = df['mpg'].values

X = X[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree = 4)
X_poly = polynomial_features.fit_transform(X)

X_poly_train = X_poly[:-20]
y_train = y[:-20]

X_poly_test = X_poly[-20:]
y_test = y[-20:]

model = LinearRegression()
model.fit(X_poly_train, y_train)
y_train_pred = model.predict(X_poly_train)
y_test_pred = model.predict(X_poly_test)

rmse_test = np.sqrt(mean_squared_error(y_test,y_test_pred))
print('\nTest RMSE: %8.15f' % rmse_test)

rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
print('\nTraining RMSE: %8.15f' % rmse)


plt.scatter(X, y, s=10)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_train_pred), key=sort_axis)
X_poly_train, y_train_pred = zip(*sorted_zip)
plt.plot(X_poly_train, y_train_pred, color='m')
plt.show()

# Logistic Regression
f = pd.read_csv("auto-mpg.csv")
df = df[df['origin'] != 'Europe']

feature_cols = ['mpg', 'displacement', 'horsepower', 'weight','acceleration']
X = df[feature_cols] 
y = df['origin'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 10)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


fig, ax = plt.subplots()
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "coolwarm" , fmt = 'g')
ax.xaxis.set_label_position("top")
plt.title('Confusion Matrix', y = 1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')

print(classification_report(y_test, y_pred))
