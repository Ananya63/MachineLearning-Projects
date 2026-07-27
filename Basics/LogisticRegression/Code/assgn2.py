"""Assgn2
Consider the above dataset. Develop a logistic regression model from the scratch using Python. 
Use cross entropy loss function and gradient descent search. 
Divide the dataset into train, validation and test in the ratio: 60:20:20 and train and test the model. 
Plot epoch vs training error and validation error."""

"""Importing the Dependencies"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

"""Data Collection and Analysis"""
# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes_dataset.csv')

# separating the data and labels
features = diabetes_dataset.drop(columns = 'Outcome', axis=1)
target = diabetes_dataset['Outcome']

"""Data Standardization"""
scaler = MinMaxScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)
features = standardized_data
target = diabetes_dataset['Outcome']

"""Train Test Split"""
# Step 1: Split into 60% training and 40% remaining (test + validation)
X_train, X_temp, Y_train, y_temp = train_test_split(features, target, test_size=0.4, random_state=42)

# Step 2: Split the remaining 40% into 20% validation and 20% test
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verifying the split ratios
print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")
X_train, X_test, Y_train, Y_test = train_test_split(features,target, test_size = 0.2, random_state=2)

# Initialize parameters
weights = np.zeros(X_train.shape[1])
learning_rate = 0.01
epochs = 1000

# Sigmoid Function
def Sigmoid(x,w):
    return 1 / (1 + np.exp(-1*(np.dot(x,w))))

# Cost Function
def compute_cost(X, y, weights):
    n = len(y)
    h = Sigmoid(X,weights)
    cost = (-1 / n) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient Descent
def gradient_descent(X, y, weights, learning_rate, epochs):
    m = X.shape[0]
    cost_history = []   
    for i in range(epochs):
        h = Sigmoid(X, weights)
        dw = (1 / m) * np.dot(X.transpose(), (h - y))
        weights -= learning_rate * dw
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

# Prediction
def predict (X, W):
    z = Sigmoid(X, W)
    z = np.where( z > 0.5, 1, 0)
    return z

# Train
weights_train, cost_history_train = gradient_descent(X_train, Y_train, weights, learning_rate, epochs)
weights_valid, cost_history_valid = gradient_descent(X_val, Y_val, weights, learning_rate, epochs)
np.savetxt('TrainSet_Weights.txt', weights_train)
np.savetxt('ValidSet_Weights.txt', weights_valid)

# Plot Training Error vs. Epochs
plt.plot(range(epochs), cost_history_train)
plt.xlabel('Epochs')
plt.ylabel('Training Error')
plt.title('Training Error vs. Epochs')
plt.show()

# Plot Validation Error vs. Epochs
plt.plot(range(epochs), cost_history_valid)
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Error')
plt.title('Validation Error vs. Epochs')
plt.show()

# accuracy score on the training data
Y_train_prediction = predict(X_train, weights_train)
training_data_accuracy = accuracy_score( Y_train, Y_train_prediction)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
Y_test_pred = predict(X_test, weights_train)
test_data_accuracy = accuracy_score( Y_test, Y_test_pred)
print('Accuracy score of the test data : ', test_data_accuracy)
