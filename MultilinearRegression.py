import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Area':[1000,1500,1800,2400,3000],
    'Bedrooms':[2,3,3,4,4],
    'Bathrooms':[1,2,2,3,3],
    'Price':[200000,300000,350000,450000,600000]
}

df = pd.DataFrame(data)

X = df[['Area','Bedrooms','Bathrooms']]
y = df['Price']

model = LinearRegression()

model.fit(X,y)

prediction = model.predict([[2000,3,2]])

print("Predicted price:",prediction)