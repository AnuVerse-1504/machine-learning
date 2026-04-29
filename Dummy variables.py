# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset (example CSV)
# Replace 'data.csv' with your file
data = pd.read_csv('data.csv')

# Step 2: Handle categorical data (Dummy Variables)
data = pd.get_dummies(data, drop_first=True)

# Step 3: Separate features and target
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column (target)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))