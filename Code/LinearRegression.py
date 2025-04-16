import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv("dataset/test.csv").dropna()
test_df = pd.read_csv("dataset/train.csv").dropna()

# Extract features and targets
X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y'].values
X_test = test_df['x'].values.reshape(-1, 1)
y_test = test_df['y'].values

# Standardize the feature
mean = X_train.mean()
std = X_train.std()
X_train_std = (X_train - mean) / std
X_test_std = (X_test - mean) / std

# Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.01
iterations = 1000
tolerance = 1e-6
m = len(y_train)
cost_history = []

# Gradient Descent
# Gradient Descent with detailed iteration output
for i in range(iterations):
    y_pred = w * X_train_std.flatten() + b
    error = y_pred - y_train

    cost = (1/(2*m)) * np.sum(error**2)
    cost_history.append(cost)

    dw = (1/m) * np.dot(error, X_train_std.flatten())
    db = (1/m) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db

    print(f"Iteration {i+1:3d} | Cost: {cost:.6f} | w: {w:.6f} | b: {b:.6f}")

    if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tolerance:
        print(f"\n✅ Converged at iteration {i+1}")
        break


# Predict on test data
y_test_pred = w * X_test_std.flatten() + b

# Evaluation metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_total)

mse = mean_squared_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
r2 = r_squared(y_test, y_test_pred)

print("\n📊 Test Set Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_test_pred, color='red', label='Predicted Line')
plt.title("Linear Regression on Test Set")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
