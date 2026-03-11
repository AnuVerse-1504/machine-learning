# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Area': [1000, 1500, 1800, 2400, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Bathrooms': [1, 2, 2, 3, 3],
    'Price': [200000, 300000, 350000, 450000, 600000]
}

# Convert to dataframe
df = pd.DataFrame(data)

# Features and target
X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

print("Predicted Prices:", prediction)

# Example prediction
new_house = [[2000, 3, 2]]  # Area, Bedrooms, Bathrooms
price = model.predict(new_house)

print("Predicted price for new house:", price)