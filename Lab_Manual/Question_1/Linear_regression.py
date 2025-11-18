import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/praveenbhagour/Desktop/Machine_Learning_Lab/Lab_Manual/Question_1/advertising.csv") 

print(data.head())

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

print("Intercept (Base Sales):", model.intercept_)
print("Coefficients:")

coef_df = pd.DataFrame({
    'Feature': ['TV', 'Radio', 'Newspaper'],
    'Coefficient': model.coef_
})

print(coef_df)

new_data = pd.DataFrame({'TV':[100], 'Radio':[20], 'Newspaper':[10]})
future_sales = model.predict(new_data)

print("\nPredicted Future Sales:", future_sales[0])
