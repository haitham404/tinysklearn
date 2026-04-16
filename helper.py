from sklearn import preprocessing as sp
from sklearn.datasets import load_iris
from tinysklearn.linear_model import LinearRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from tinysklearn.metrics.regression import mean_squared_error
from tinysklearn.metrics.regression import r2_score
from tinysklearn.preprocessing import   PolynomialFeatures

data = load_iris()
X = data.data
y = data.target     

poly=PolynomialFeatures(degree=1)
X_new=poly.fit_transform(X)




model = LinearRegression()
model.fit(X_new, y)
y_pred = model.predict(X_new)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)
r2=r2_score(y,y_pred)
print("R2 score",r2)




