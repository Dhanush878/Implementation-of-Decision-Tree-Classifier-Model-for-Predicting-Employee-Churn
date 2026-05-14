# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the employee dataset using pandas.
2.Convert the categorical salary column into numerical values using Label Encoding.
3.Select input features and target variable (left).
4.Split the dataset into training and testing sets.
5.Train the Decision Tree Classifier using the entropy criterion and predict test results.
6.Calculate model accuracy and predict whether a new employee will leave or not.. 

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DHANUSH M D
RegisterNumber:212224100011 


import pandas as pd
data = pd.read_csv(r"C:\Users\acer\Downloads\Employee (1).csv")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
X = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
X.head()
y=data["left"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
<img width="1260" height="101" alt="589379669-843f1abc-ac36-42d6-9e53-1b52659c31fd" src="https://github.com/user-attachments/assets/a8f4cc33-777e-48dd-936b-2ce8d9499340" />
<img width="993" height="86" alt="589380166-e0e05df2-3e76-48e7-aeee-8f461c3acf9c" src="https://github.com/user-attachments/assets/1d350a6c-fb15-4088-b71e-55b1ee723e51" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
