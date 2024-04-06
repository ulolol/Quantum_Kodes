"""
    Simple Quantum Perceptron with Adam Optimizer for
    Parameter Training 
    
    Author: Vidish 
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.primitives import BackendSampler

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
    for i, weight in enumerate(weights):
        circuit.rz(weight, qr[i])
    circuit.rz(bias, qr[n-1])

    #Apply Pauli-X gate to last qubit if sum of weights is less than threshold
    circuit.x(qr[n-1]).c_if(cr, 1)

    #Measure the last qubit
    circuit.measure(qr[n-1], cr[0])

    return circuit

def adam_optimizer(parameters, gradients, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    This function performs parameter optimization using the Adam algorithm.

    Args:
        parameters (np.array): The parameters to optimize.
        gradients (np.array): The gradients of the parameters.
        v (np.array): The first moment vector in Adam.
        s (np.array): The second moment vector in Adam.
        t (int): The timestep for the Adam optimizer.
        learning_rate (float): The learning rate for Adam.
        beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
        epsilon (float, optional): A small constant for numerical stability. Defaults to 1e-8.

    Returns:
        tuple: The updated parameters, first moment vector, and second moment vector.
    """

    v = beta1 * v + (1 - beta1) * gradients
    s = beta2 * s + (1 - beta2) * np.square(gradients)
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    parameters -= learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return parameters, v, s

#Example usage
inputs = [1, 0, 1]
weights = [0, 0, 0]
bias = 0
threshold = 2

circuit = perceptron_circuit(inputs, weights, bias, threshold)

#Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
new_circuit = transpile(circuit, backend)
sampler = BackendSampler(backend)
job = sampler.run(new_circuit)
result = job.result()

#Get the result
print(f"Output: {result}")

#Run the perceptron with Adam optimizer
t = 1
learning_rate = 0.01
v = np.zeros_like(weights)
s = np.zeros_like(weights)
for _ in range(10):
    gradients = np.array([0.1, 0.2, 0.3])
    weights, v, s = adam_optimizer(weights, gradients, v, s, t, learning_rate)
    print("Updated weights:", weights)
