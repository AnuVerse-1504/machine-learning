import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Hours':[1,2,3,4,5,6,7,8],
    'Marks':[35,40,50,55,65,70,80,88]
}

df = pd.DataFrame(data)

# Input and Output
X = df[['Hours']]
y = df['Marks']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Model
model = LinearRegression()

# Training
model.fit(X_train,y_train)

# Prediction
predicted = model.predict(X_test)

print("Actual Marks:",y_test.values)
print("Predicted Marks:",predicted)

# Predict new value
hours = [[9]]
prediction = model.predict(hours)

print("Predicted Marks for 9 hours study:",prediction)

# Graph
plt.scatter(X,y,color="blue")
plt.plot(X,model.predict(X),color="red")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()