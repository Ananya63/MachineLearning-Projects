'''Write a python program to implement the BP algorithm with momentum for ANN with more than 1 hidden layers.  Consider two different situations (1) all nodes are sigmoid.(2) hidden nodes are ReLu and output nodes are  sigmoid. Design the model in such a way that one can vary hidden layers and nodes per hidden layer. Train the models with diabetes datasets given to you earlier.'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Load and preprocess the diabetes dataset
data = pd.read_csv("/home/dst-fist/Desktop/ML_LAB/diabetes_dataset.csv")
X = data.drop(columns=['Outcome'])
y = data['Outcome'].values.reshape(-1, 1)

# Standardizing the data (important for neural networks)
scaler_X = StandardScaler()  # For feature scaling
X_scaled = scaler_X.fit_transform(X)

# Standardizing the target variable 'y'
scaler_y = StandardScaler()  # For target variable scaling
y_scaled = scaler_y.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# ReLU and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Function to initialize the weights for the network
def initialize_weights(layer_sizes):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
        bias = np.zeros((1, layer_sizes[i+1]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases

# Feedforward function
def feedforward(X, weights, biases, activations):
    output = X
    outputs = [output]
    for i in range(len(weights)):
        z = np.dot(output, weights[i]) + biases[i]
        if activations[i] == 'sigmoid':
            output = sigmoid(z)
        elif activations[i] == 'relu':
            output = relu(z)
        outputs.append(output)
    return outputs

def backpropagate(X, y, weights, biases, activations, learning_rate, momentum, prev_d_weights, prev_d_biases):
    m = X.shape[0]
    outputs = feedforward(X, weights, biases, activations)
    
    # Calculate the error at the output layer
    output_error = outputs[-1] - y
    
    d_weights = []
    d_biases = []

    # Go backward through the network
    for i in range(len(weights) - 1, -1, -1):
        # Adjust for the correct activation function for each layer
        if i < len(activations):  # Ensure we are within the bounds of the activations list
            if activations[i] == 'sigmoid':
                d_activation = sigmoid_derivative(outputs[i + 1]) * output_error
            elif activations[i] == 'relu':
                d_activation = relu_derivative(outputs[i + 1]) * output_error
        else:
            d_activation = output_error
        
        d_weight = np.dot(outputs[i].T, d_activation) / m
        d_bias = np.sum(d_activation, axis=0, keepdims=True) / m
        
        # Update the error to propagate backward to the previous layer
        if i > 0:
            output_error = np.dot(d_activation, weights[i].T)
        
        # Apply momentum to the gradients
        if prev_d_weights[i] is None:
            prev_d_weights[i] = np.zeros_like(d_weight)
            prev_d_biases[i] = np.zeros_like(d_bias)
        
        d_weights.append(d_weight + momentum * prev_d_weights[i])
        d_biases.append(d_bias + momentum * prev_d_biases[i])

        prev_d_weights[i] = d_weight
        prev_d_biases[i] = d_bias

    # Reverse the list of gradients since we were going backward
    d_weights = d_weights[::-1]
    d_biases = d_biases[::-1]
    
    return d_weights, d_biases, prev_d_weights, prev_d_biases

# Train the neural network
def train(X_train, y_train, layer_sizes, activations, learning_rate=0.01, momentum=0.9, epochs=1000):
    weights, biases = initialize_weights(layer_sizes)
    prev_d_weights = [None] * (len(layer_sizes) - 1)
    prev_d_biases = [None] * (len(layer_sizes) - 1)
    
    for epoch in range(epochs):
        d_weights, d_biases, prev_d_weights, prev_d_biases = backpropagate(X_train, y_train, weights, biases, activations, learning_rate, momentum, prev_d_weights, prev_d_biases)

        # Update the weights and biases
        for i in range(len(weights)):
            weights[i] -= learning_rate * d_weights[i]
            biases[i] -= learning_rate * d_biases[i]

        if epoch % 100 == 0:
            loss = np.mean((feedforward(X_train, weights, biases, activations)[-1] - y_train) ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return weights, biases

# Predict with the trained network
def predict(X, weights, biases, activations):
    return feedforward(X, weights, biases, activations)[-1]

# Define layer sizes and activation functions
layer_sizes = [X_train.shape[1], 64, 32, 1]  # Input layer -> 2 hidden layers -> output layer
activation1 = ['relu', 'relu', 'sigmoid']  # Hidden layers use ReLU, output uses sigmoid
activation2 = ['sigmoid', 'sigmoid', 'sigmoid']  # All layers use sigmoid

# Train the network
weights1, biases1 = train(X_train, y_train, layer_sizes, activation1, learning_rate=0.01, momentum=0.9, epochs=1000)
weights2, biases2 = train(X_train, y_train, layer_sizes, activation2, learning_rate=0.01, momentum=0.9, epochs=1000)

# Make predictions on the test set
prediction1 = predict(X_test, weights1, biases1, activation1)
prediction2 = predict(X_test, weights2, biases2, activation2)

# Inverse transform predictions and test labels to original scale
predictions_original_1 = scaler_y.inverse_transform(prediction1)
predictions_original_2 = scaler_y.inverse_transform(prediction2)
y_test_original = scaler_y.inverse_transform(y_test)

predictions_binary1 = (predictions_original_1 > 0.5).astype(int)  # Convert probabilities to binary class labels (0 or 1)
predictions_binary2 = (predictions_original_2 > 0.5).astype(int)

# Display some predictions and the corresponding true values
print("Predictions where Hidden layers use ReLU, output uses sigmoid:", predictions_binary1[:5])
print("Predictions where All layers use sigmoid:", predictions_binary2[:5])
print("True values:", y_test_original[:5])

# Calculate accuracy
accuracy1 = accuracy_score(y_test_original, predictions_binary1)
accuracy2 = accuracy_score(y_test_original, predictions_binary2)

# Display the accuracy
print(f"Accuracy where Hidden layers use ReLU, output uses sigmoid: {accuracy1 * 100:.2f}%")
print(f"Accuracy where All layers use sigmoid: {accuracy2 * 100:.2f}%")
