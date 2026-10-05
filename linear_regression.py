import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv("data/housing.csv")

print("Dataset shape:", data.shape)
print(data.head())


# Remove missing values
data = data.dropna()


# Features and target
features = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "latitude",
    "longitude"
]

X = data[features].values.astype(float)
y = data["median_house_value"].values.astype(float)


# Shuffle the data
np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]


# Split into training and testing data
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Standardize the features
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# Initialize weights and bias
w = np.zeros(X_train.shape[1])
b = 0.0

learning_rate = 0.01
epochs = 5000

loss_history = []


# Train the model using gradient descent
for epoch in range(epochs):

    # Prediction
    y_pred = X_train @ w + b

    # Difference between prediction and actual value
    error = y_pred - y_train

    # Mean Squared Error
    mse = np.mean(error ** 2)
    loss_history.append(mse)

    # Calculate gradients
    dw = (2 / len(X_train)) * (X_train.T @ error)
    db = 2 * np.mean(error)

    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: MSE = {mse:.2f}")


# Make predictions
train_predictions = X_train @ w + b
test_predictions = X_test @ w + b


# Evaluation metrics
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot


print("\nResults")

print("\nTraining:")
print("MSE:", mse(y_train, train_predictions))
print("MAE:", mae(y_train, train_predictions))
print("R2 :", r2_score(y_train, train_predictions))

print("\nTesting:")
print("MSE:", mse(y_test, test_predictions))
print("MAE:", mae(y_test, test_predictions))
print("R2 :", r2_score(y_test, test_predictions))


# Learned weights
print("\nLearned weights:")

for feature, weight in zip(features, w):
    print(f"{feature}: {weight:.4f}")

print("Bias:", b)


# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training Loss")
plt.grid(True)

plt.show()


# Actual vs predicted values
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.3
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted")

plt.grid(True)

plt.show()
