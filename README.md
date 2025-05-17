# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Jayaseelan U
RegisterNumber:  212223220039
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

data = pd.read_csv("Salary.csv")
print(data.head(), "\n")
print(data.info(), "\n")
print(data.isnull().sum(), "\n")

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head(), "\n")

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor(random_state=2)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(mse)
print(r2)

print(dt.predict([[5, 6]])[0])
```


## Output:
## Data.Head():
![91](https://github.com/user-attachments/assets/d2dc9ba2-ebe0-460a-8fe8-6c42ae9c3047)


## data.isnull().sum():
![92](https://github.com/user-attachments/assets/ca41f196-311a-42f5-a280-1503d725c401)

## data.head() for salary:

![93](https://github.com/user-attachments/assets/41bdac48-7e8d-4954-a5ed-fe7154b7d0d2)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
