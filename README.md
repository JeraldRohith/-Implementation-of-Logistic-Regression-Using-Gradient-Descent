# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np

# Load and preprocess the data
data = pd.read_csv("Placement_Data-2.csv")

# Map status: Placed -> 1, Not Placed -> 0
data = data.dropna(subset=['status'])  # Remove rows with missing status
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Select numeric features
features = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
X = data[features].values
y = data['status'].values

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add intercept
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression with gradient descent
def logistic_regression(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    for epoch in range(epochs):
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
        error = y_pred - y
        gradient = np.dot(X.T, error) / n_samples
        weights -= lr * gradient
        # Optional: print loss occasionally
        if epoch % 100 == 0:
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            print(f'Epoch {epoch}: Loss={loss:.4f}')
    return weights

# Fit the model
weights = logistic_regression(X, y, lr=0.05, epochs=2000)

# Predict function
def predict(X, weights):
    probs = sigmoid(np.dot(X, weights))
    return (probs >= 0.5).astype(int)

# Evaluate
y_pred = predict(X, weights)
accuracy = np.mean(y_pred == y)
print(f'Accuracy: {accuracy * 100:.2f}%')

```

## Output:


<img width="849" height="641" alt="image" src="https://github.com/user-attachments/assets/1e15ec98-0efe-4c7a-8b29-938b35aaf5cc" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

