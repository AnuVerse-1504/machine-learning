import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = {
    'Study_Hours':[1,2,3,4,5,6],
    'Attendance':[50,60,65,70,80,90],
    'Result':[0,0,0,1,1,1]  # 0=Fail, 1=Pass
}

df = pd.DataFrame(data)

X = df[['Study_Hours','Attendance']]
y = df['Result']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Prediction:",prediction)