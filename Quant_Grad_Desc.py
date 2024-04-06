"""
Quantum Gradient Descent

This script implements a Quantum Gradient Descent model using Qiskit, 
a Python library for quantum computing. 
The model uses a quantum perceptron, a type of quantum neural network, 
as the base estimator for gradient descent.
It includes a custom estimator class 'QuantumGradientDescent' 
that wraps around a quantum gradient descent function.

The script includes the following features:
    1. Data generation using sklearn's make_classification function.
    2. Data splitting into training and testing sets.
    3. Training a gradient descent model with early stopping.
    4. Making predictions on the test set and calculating the accuracy of the model.
    5. Bayesian hyperparameter tuning using skopt's BayesSearchCV function.

Quantum elements of the script include:
    1. Quantum Perceptron: The script uses a quantum circuit to represent 
    a perceptron model. The quantum circuit applies a Hadamard gate to all 
    qubits, applies weights and bias as rotations, and measures the last qubit.

    2. Quantum Circuit Execution: The quantum circuits are executed using 
    Qiskit's Aer simulator. The simulator runs the quantum circuits and 
    returns the measurement counts.

The script also includes classical machine learning techniques, such as :
    1. Gradient Descent for training the model 
    2. Bayesian optimization for hyperparameter tuning

Classes:
    QuantumGradientDescent: A custom estimator for Quantum Gradient Descent.

Functions:
    perceptron_circuit: Generates a quantum circuit for a perceptron model.
    execute_quantum_circuit: Executes the quantum circuit to generate an
        expectation value for the gradient
    gradient_descent: Performs gradient descent on a quantum perceptron model.

Codeflow:
The script first generates a simple dataset and splits it into training 
and testing sets. It then trains a gradient descent model on the training 
set, with early stopping if the validation loss does not improve for a 
certain number of epochs. It makes predictions on the test set and 
calculates the accuracy of the model. 
If the '--tune' flag is given when running the script, it also performs 
Bayesian hyperparameter tuning.

Usage:
    python Quant_Grad_Desc.py [--tune]

Author: Vidish Srivastava
"""


import argparse

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

from skopt import BayesSearchCV
from skopt.space import Real, Integer

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer


#Create the parser
parser = argparse.ArgumentParser(description='Run Bayesian hyperparameter tuning.')

#Add an argument
parser.add_argument('--tune', action='store_true',
                    help='a flag to enable Bayesian hyperparameter tuning')

#Parse the arguments
args = parser.parse_args()

#Custom estimator class
class QuantumGradientDescent(BaseEstimator, ClassifierMixin):
    """
    Custom estimator for Quantum Gradient descent.

    This estimator is a wrapper around a quantum gradient descent function. 
    It inherits from sklearn's BaseEstimator and ClassifierMixin.

    Args:
        num_estimators: int, optional (default=10)
            The number of descent stages to perform. 
            Gradient descent is fairly robust to over-fitting so a large 
            number usually results in better performance.

        learning_rate: float, optional (default=0.001)
            Learning rate shrinks the contribution of each classifier by learning_rate. 
            There is a trade-off between learning_rate and num_estimators.

    Attributes:
        weights_: array of shape = [n_features]
            The weights of the quantum perceptron circuit.

        bias_: float
            The bias of the quantum perceptron circuit.
    """

    def __init__(self, num_estimators=70, learning_rate=0.009):
        self.num_estimators = num_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Fit the gradient descent model.

        Args:
            X: {array-like, sparse matrix} of shape = [n_samples, n_features]
                The training input samples.

            y: array-like of shape = [n_samples]
                The target values (class labels in classification).

        Returns:
            self: object
                Returns self.
        """
        self.weights_, self.bias_ = gradient_descent(
            X, y, self.num_estimators, self.learning_rate
        )
        return self

    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is computed by running the quantum circuit 
        with the optimized weights and bias, and taking the most common outcome.

        Args:
            X: {array-like, sparse matrix} of shape = [n_samples, n_features]
                The input samples.

        Returns:
            predictions: array of shape = [n_samples]
                The predicted classes.
        """
        predictions = []
        for x in X:
            backend = Aer.get_backend('qasm_simulator')
            circuit = perceptron_circuit(x, self.weights_, self.bias_, 0)
            transpiled_circuit = transpile(circuit, backend)
            job = backend.run(transpiled_circuit)
            counts = job.result().get_counts()
            y_pred = max(counts, key=counts.get)
            predictions.append(int(y_pred))
        return predictions


def perceptron_circuit(inputs, weights, bias, threshold):
    """
    This function generates a quantum circuit for a perceptron model.

    Args:
        inputs (list): The input values.
        weights (list): The weights of the perceptron.
        bias (float): The bias of the perceptron.
        threshold (float): The threshold for the perceptron activation.

    Returns:
        QuantumCircuit: A quantum circuit representing the perceptron.
    """

    n = len(inputs)
    qr = QuantumRegister(n)
    cr = ClassicalRegister(1)
    circuit = QuantumCircuit(qr, cr)

    #Apply Hadamard gate to all qubits
    circuit.h(qr)

    #Apply weights and bias
    for i in range(n):
        circuit.rz(weights[i], qr[i])
    circuit.rz(bias, qr[n-1])

    #Apply Pauli-X gate to last qubit if sum of weights is less than threshold
    circuit.x(qr[n-1]).c_if(cr, 1)

    #Measure the expectation value of the Z operator on the last qubit
    circuit.h(qr[n-1])
    circuit.measure(qr[n-1], cr[0])

    return circuit

def execute_quantum_circuit(inputs, weights, bias, threshold):
    """
    This function executes a quantum circuit for a perceptron model 
    and returns the expectation value.

    Args:
        inputs (list): The input values.
        weights (list): The weights of the perceptron.
        bias (float): The bias of the perceptron.
        threshold (float): The threshold for the perceptron activation.

    Returns:
        float: The expectation value of the Z operator on the last qubit.
    """

    #Create the quantum circuit
    circuit = perceptron_circuit(inputs, weights, bias, threshold)

    #Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit, shots=1000)
    result = job.result()

    #Calculate the expectation value
    counts = result.get_counts()
    expectation_value = (counts.get('0', 0) - counts.get('1', 0)) / sum(counts.values())

    return expectation_value

def gradient_descent(X, y, num_estimators, learning_rate):
    """
    This function performs gradient descent on a quantum perceptron model.

    Args:
        X (np.array): The input data.
        y (np.array): The target data.
        num_estimators (int): The number of descent stages to perform.
        learning_rate (float): The learning rate for gradient descent.

    Returns:
        tuple: The final weights and bias for the perceptron model.
    """

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    EPOCH = 0

    #Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_loss = np.inf
    patience = 5
    no_improvement = 0

    #Create an instance of QuantumGradientDescent
    model = QuantumGradientDescent(num_estimators=num_estimators, learning_rate=learning_rate)

    #Perform gradient descent
    for _ in range(num_estimators):
        for i in range(n_samples):
            #Print the current epoch
            print(f"[EPOCH: {EPOCH}", end=", "); EPOCH += 1

            #Create and run the quantum circuit
            circuit = perceptron_circuit(X[i], weights, bias, 0)
            backend = Aer.get_backend('qasm_simulator')
            transpiled_circuit = transpile(circuit, backend)
            job = backend.run(transpiled_circuit)
            counts = job.result().get_counts()
            
            #Calculate the gradient of the loss function
            y_pred = execute_quantum_circuit(X[i], weights, bias, 0)
            if y_pred == 0:
                gradient = y[i]
            elif y_pred == 1:
                gradient = - (1 - y[i])
            else:
                gradient = -(y[i] / y_pred - (1 - y[i]) / (1 - y_pred))
            print(f"Current Grad: {gradient}")

            #Update the weights and bias
            weights -= learning_rate * gradient
            print(f"Current weights: {weights}")
            bias -= learning_rate * (y[i] - int(y_pred))
            print(f"Current bias: {bias}]")

        #Set the weights and bias in the model
        model.weights_ = weights
        model.bias_ = bias

        #Calculate the loss on the validation set
        val_predictions = model.predict(X_val)
        val_loss = np.mean((y_val - val_predictions) ** 2)

        #If the validation loss improved, save the model and reset the patience
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = weights
            best_bias = bias
            no_improvement = 0
        else:
            no_improvement += 1

        #If the validation loss didn't improve for 'patience' epochs, stop training
        if no_improvement >= patience:
            print("Early stopping due to no improvement")
            break

    return best_weights, best_bias


#Generate a simple dataset
X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the gradient descent model
num_estimators = 70
learning_rate = 0.09933171128208856
weights, bias = gradient_descent(X_train, y_train, num_estimators, learning_rate)

#Make predictions on the test set
predictions = []
for x in X_test:
    backend = Aer.get_backend('qasm_simulator')
    circuit = perceptron_circuit(x, weights, bias, 0)
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit)
    counts = job.result().get_counts()
    y_pred = max(counts, key=counts.get)
    predictions.append(int(y_pred))

#Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

#Run Bayesian hyperparameter tuning if the --tune flag is given
if args.tune:
    #Define the hyperparameter space
    param_space = {
        'num_estimators': Integer(10, 100),
        'learning_rate': Real(0.001, 0.1)
    }

    #Initialize a cross-validation fold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    #Initialize the optimizer
    opt = BayesSearchCV(
        QuantumGradientDescent(),
        param_space,
        cv=cv,
        n_jobs=-1,
        n_iter=50,
    )

    #Fit the optimizer
    opt.fit(X_train, y_train)

    print("Best parameters found: ", opt.best_params_)
    print("Best accuracy found: ", opt.best_score_)


##xxxxxxxxxx----------xxxxxxxxxx
##Possible hyperparameters
#num_estimators_list = [10, 50, 100]
#learning_rate_list = [0.001, 0.01, 0.1]
#
#best_accuracy = 0
#best_hyperparameters = None
#
##Iterate over all possible combinations of hyperparameters
#for num_estimators in num_estimators_list:
    #for learning_rate in learning_rate_list:
        #weights, bias = gradient_descent(X_train, y_train, num_estimators, learning_rate)
#   
        ##Make predictions on the test set
        #predictions = []
        #for x in X_test:
            #backend = Aer.get_backend('qasm_simulator')
            #circuit = perceptron_circuit(x, weights, bias, 0)
            #transpiled_circuit = transpile(circuit, backend)
            #job = backend.run(transpiled_circuit)
            #counts = job.result().get_counts()
            #y_pred = max(counts, key=counts.get)
            #predictions.append(int(y_pred))
#
        ##Calculate the accuracy of the model
        #accuracy = accuracy_score(y_test, predictions)
        #print(f"Accuracy with num_estimators={num_estimators}, learning_rate={learning_rate}: {accuracy}")
#
        ##Update best accuracy and hyperparameters
        #if accuracy > best_accuracy:
            #best_accuracy = accuracy
            #best_hyperparameters = (num_estimators, learning_rate)
#
#print(f"Best accuracy: {best_accuracy}")
#print(f"Best hyperparameters: num_estimators={best_hyperparameters[0]}, learning_rate={best_hyperparameters[1]}")
#
