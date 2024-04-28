"""
    v3 goes back to basics and implements an iterative perceptron 
    neural network using the IterativePerceptron class which 
    initializes with specified numbers of inputs, hidden neurons, 
    and outputs.

    Trying to figure out what is more robust compared to v2, which
    used the theta angle as an optimization modifier on the
    Quantum Circuit. 

    Also trying to use concurrency during training to 
    improve performance.

    Author: Vidish Srivastava
"""
import os
import numpy as np
import concurrent.futures
from qiskit import QuantumCircuit
from qiskit_aer import Aer

class IterativePerceptron:
    """
    A neural network implementation using iterative perceptron algorithm.
    """
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int) -> None:
        """
        Initializes the IterativePerceptron class with specified number of inputs, hidden neurons and outputs.

        Args:
            num_inputs (int): The number of input variables.
            num_hidden (int): The number of hidden neurons.
            num_outputs (int): The number of output variables.

        Returns:
            None
        """
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.weights1 = np.random.rand(num_inputs, num_hidden)
        self.weights2 = np.random.rand(num_hidden, num_outputs)
        self.bias1 = np.zeros((num_hidden,))
        self.bias2 = np.zeros((num_outputs,))


    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for a given input.

        Args:
            x (numpy array): The input to be passed through the network.

        Returns:
            numpy array: The output of the network.
        """
        #Quantum Circuit for hidden layer
        qc = QuantumCircuit(1, self.num_hidden)
        qc.ry(np.pi / 2, 0)
        qc.barrier()
        qc.measure_all()
        job = Aer.get_backend('qasm_simulator').run(qc, shots=10**6).result()
        counts = job.get_counts(qc)
        hidden_layer_prob = np.array([counts[k] for k in counts]) / len(counts)

        #Clip probabilities to the nearest multiple of 0.01
        hidden_layer_prob = np.clip(hidden_layer_prob, a_min=0, a_max=1) * 0.01

        #Convert hidden layer to numpy array (binary representation)
        hidden_layer = np.array(['0' if p < 0.5 else '1' for p in hidden_layer_prob])

        #Convert binary representation to float type
        hidden_layer = np.float32([1.0 if bit == '1' else 0.0 for bit in hidden_layer])

        #Compute output layer
        output_layer = np.dot(hidden_layer, self.weights2) + self.bias2
        return output_layer


    def train_threaded(self, X: np.ndarray, y: np.ndarray, learning_rate: float, num_iterations: int) -> None:
        """
        Trains the network using threaded processing.

        Args:
            X (numpy array): The input data to be trained on.
            y (numpy array): The output labels for the training data.
            learning_rate (float): The rate at which weights are updated during training.
            num_iterations (int): The number of training iterations.

        Returns:
            None
        """
        max_workers=os.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            futures = []
            for _ in range(num_iterations):
                print(f"\n Epoch: {_}")
                for x, target in zip(X, y):
                    futures.append(executor.submit(self._train_single_sample, x, target, learning_rate))
                
                for future in concurrent.futures.as_completed(futures):
                    self.weights1 += learning_rate * future.result()[0]
                    self.bias1 += learning_rate * future.result()[1]
                    self.weights2 += learning_rate * future.result()[2]
                    self.bias2 += learning_rate * future.result()[3]
                    print(f"Weight1: {self.weights1}, Weight2: {self.weights2}")
                    print(f"Bias1: {self.bias1}, Bias2: {self.bias2} \n")


    def _train_single_sample(self, x: np.ndarray, target: np.ndarray, learning_rate: float) -> tuple:
        """
        Trains the network for a single input sample.

        Args:
            x (numpy array): The input sample.
            target (numpy array): The expected output label for the input sample.
            learning_rate (float): The rate at which weights are updated during training.

        Returns:
            tuple: A tuple containing the updates to the weights and biases.
        """
        output = self.forward_pass(x)
        error = target - output
        return (x * error * learning_rate, error * learning_rate, output * error * learning_rate, error * learning_rate)

                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for a given input and returns the predicted output.

        Args:
            X (numpy array): The input to be passed through the network.

        Returns:
            numpy array: The predicted output of the network.
        """
        return [self.forward_pass(x) for x in X]


#Test00
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
ip = IterativePerceptron(2, 2, 1)
ip.train_threaded(X, y, learning_rate=0.1, num_iterations=10)
print(f"Prediction: {ip.predict(X)}")
