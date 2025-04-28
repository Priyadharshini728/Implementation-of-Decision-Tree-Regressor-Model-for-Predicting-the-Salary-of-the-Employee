# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2. Calculate the null values present in the dataset and apply label encoder.

3. Determine test and training data set and apply decison tree regression in dataset.

4. Calculate Mean square error,data prediction and r2.


## Program:

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: PRIYADHARSHINI P

RegisterNumber:212224040252  
*/

```py

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

### DATA HEAD:

![image](https://github.com/user-attachments/assets/5efa4cf4-ed93-4e0b-b880-7f85a3014223)


### DATA INFO:

![image](https://github.com/user-attachments/assets/0b98b28a-e8b0-49ee-8ae2-0f41f7a1b31c)


### ISNULL() AND SUM():

![image](https://github.com/user-attachments/assets/579aa281-7d5a-4f0b-9ce9-29bdf3a395cf)


### DATA HEAD FOR SALARY:

![image](https://github.com/user-attachments/assets/0e391103-f61a-4f31-bfcc-356973aadbc1)


### MEAN SQUARED ERROR:

![image](https://github.com/user-attachments/assets/8abf08e8-7fa7-42e1-bb3a-3892ce9f6ee8)



### R2 VALUE:

![image](https://github.com/user-attachments/assets/3d0531b3-727d-4265-bb0c-068631a0143b)



### DATA PREDICTION:


![image](https://github.com/user-attachments/assets/b7f4a9cb-b6c9-4586-a249-54fb92fc074b)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
