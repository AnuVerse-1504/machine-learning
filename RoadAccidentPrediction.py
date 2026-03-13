import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Traffic_Volume': [200, 500, 300, 700, 1000, 650],
    'Weather_Condition': [1, 2, 1, 3, 2, 3],  # 1=Clear, 2=Rain, 3=Fog
    'Average_Speed': [60, 40, 55, 30, 35, 45],
    'Accidents': [2, 5, 3, 7, 9, 6]
}

df = pd.DataFrame(data)

X = df[['Traffic_Volume','Weather_Condition','Average_Speed']]
y = df['Accidents']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Predicted accidents:",prediction)