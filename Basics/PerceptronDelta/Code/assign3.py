"""Assgn3
Assignment: Consider a single linear neural node and use perceptron rule and delta rule to train it. Train the model upto 100 epoch and display error after each epoch.
Print the learned weight values.
For each case plot the data in feature space and   display the separator.
"""
import numpy as np
import matplotlib.pyplot as plt

# Training dataset1
X = np.array([
    [0.3, 0.2],
    [0.8, 0.9],
    [0.1, 0.8],
    [0.95, 0.9],
    [0.9, 0.4],
    [0.7, 0.8],
    [0.4, 0.3],
    [0.2, 0.85],
    [0.82, 0.93],
    [0.75, 0.25]
])
y1 = np.array([-1, 1, -1, 1, -1, 1, -1, -1, 1, -1])

# Training dataset2
y2 = np.array([-1, -1, 1, -1, 1, -1, -1, 1, -1, 1])

# Parameters
c= int(input("Training set 1 or 2?: "))
if(c==1):
    y=y1
else:
    y=y2
learning_rate = 0.1
epochs = 100

# Initialize weights and bias
weights_perceptron = np.random.rand(2)
bias_perceptron = np.random.rand(1)
weights_delta = np.random.rand(2)
bias_delta = np.random.rand(1)

# Training with Perceptron rule
print("\nTraining with Perceptron rule")
perceptron_errors = []
for epoch in range(epochs):
    correct_predictions = 0
    for i in range(len(X)):
        # Perceptron prediction
        activation = np.dot(X[i], weights_perceptron) + bias_perceptron
        y_pred = 1 if activation >= 0 else -1  # Threshold activation
        # Perceptron weight update if misclassified
        error = y[i] - y_pred
        if error != 0:
            weights_perceptron += learning_rate * error * X[i]
            bias_perceptron += learning_rate * error
        else:
            correct_predictions += 1  # Count correctly classified samples
    # Calculate accuracy and total error for this epoch
    accuracy = correct_predictions / len(X)
    total_error = 1 - accuracy
    #total_error += abs(error)
    print(f"Perceptron Error after epoch {epoch+1} : {total_error}")
    perceptron_errors.append(total_error)

# Training with Delta rule
print("\nTraining with Delta rule")
delta_errors = []
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        # Delta rule prediction
        y_pred = np.dot(X[i], weights_delta) + bias_delta
        # Delta rule weight update
        error = y[i] - y_pred
        weights_delta += learning_rate * error * X[i] 
        bias_delta += learning_rate * error 
        total_error += error**2
    print(f"Delta Error after epoch {epoch}: {total_error}")
    delta_errors.append(total_error/len(X))

# Final learned weights and biases
print("weights_perceptron",weights_perceptron)
print("bias_perceptron",bias_perceptron)
#print("perceptron_errors",perceptron_errors)
print("\n")
print("weights_delta",weights_delta)
print("bias_delta",bias_delta)
#print("delta_errors",delta_errors)

# Plot Perceptron Errors per epoch
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), perceptron_errors, marker='o', color='b', label='Perceptron Error')
plt.title('Perceptron Error per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()

# Plot Delta Rule Errors per epoch
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), delta_errors, marker='o', color='r', label='Delta Error')
plt.title('Delta Rule Error per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()

# Plotting decision boundaries and data points
def plot_decision_boundary(X, y, weights, bias, title="Decision Boundary"):
    plt.figure(figsize=(6, 4))
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o')
    
    # Calculate the decision boundary
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Plot decision boundary
    x_values = np.array([x_min, x_max])
    y_values = -(weights[0] * x_values + bias) / weights[1]
    plt.plot(x_values, y_values, color='black', linewidth=2, label="Decision Boundary")
    plt.title(title)
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.show()

# Plotting the decision boundary for Perceptron
plot_decision_boundary(X, y, weights_perceptron, bias_perceptron, "Perceptron Decision Boundary")

# Plotting the decision boundary for Delta Rule
plot_decision_boundary(X, y, weights_delta, bias_delta, "Delta Rule Decision Boundary")
