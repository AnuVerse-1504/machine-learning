# Machine-learning
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
x=np.array([1,2,3,4,5]).reshape(-1,1)
y=np.array([2,4,6,8,10])

model=LinearRegression()

model.fit(x,y)

y_pred=model.predict(x)

print("Slope(m):"model.coef_)
print("intercept(c):",model.intercept_)

plt.scatter(x,y,color='blue',label='Actual Data')
plt.plot(x,y_pred,color='red',label='Regression Line')
plt.xlabel("x")
ply.ylabel("y")
plt.legend()
plt.show()












