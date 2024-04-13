"""
    This script implements a parametrized quantum perceptron optimization 
    using Rotation Angle Theta. 
    It creates a single qubit quantum circuit, applies a rotation gate, 
    measures the output, and then uses a gradient descent algorithm to 
    iteratively update the rotation angle to minimize the difference 
    between the output and the target value. 
    The training process includes an adaptive learning rate and early 
    stopping mechanism.

Author: Vidish Srivastava
"""
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from qiskit.primitives import BackendSampler
import numpy as np


def create_parameterized_circuit():
    """
    Creates a parameterized quantum circuit with a rotation and measurement, 
    and returns the circuit and the Parameter object for the rotation angle.
    """
    theta = Parameter('θ')
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    qc.ry(theta, qr[0])
    qc.measure(qr, cr)
    return qc, theta


def execute_with_theta(qc, theta_param, theta_value):
    """
    Binds the theta_value to the theta_param in the quantum circuit, executes the circuit, 
    and returns the probability of measuring |1>.
    """
    backend = Aer.get_backend('qasm_simulator')
    bound_circuit = qc.assign_parameters({theta_param: theta_value})
    transpiled_circuit = transpile(bound_circuit, backend)
    job = backend.run(transpiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts(transpiled_circuit)
    probability_of_one = counts.get('1', 0) / 1024
    return probability_of_one


def compute_gradient(qc, theta_param, theta_value, epsilon=0.01):
    """
    Computes the gradient of the output probability with respect to theta_value 
    using finite differences.
    """
    prob_plus = execute_with_theta(qc, theta_param, theta_value + epsilon)
    prob_minus = execute_with_theta(qc, theta_param, theta_value - epsilon)
    gradient = (prob_plus - prob_minus) / (2 * epsilon)
    return gradient


def quantum_perceptron_training(initial_theta=np.pi / 4, base_lr=0.1, 
                                epochs=100, early_stopping_threshold=0.01, 
                                early_stopping_patience=10):
    """
    Trains a quantum perceptron using a parameterized quantum circuit, 
    and returns the optimized theta value. The training process includes 
    an adaptive learning rate and early stopping mechanism.
    """
    qc, theta = create_parameterized_circuit()
    
    theta_value = initial_theta
    last_loss = None
    loss_increase_count = 0
    adaptive_lr = base_lr

    for epoch in range(epochs):
        output = execute_with_theta(qc, theta, theta_value)
        loss = (1 - output) ** 2
        grad = compute_gradient(qc, theta, theta_value)

        if last_loss is not None and loss < last_loss:
            adaptive_lr *= 1.17
        else:
            adaptive_lr *= 0.97

        theta_value -= adaptive_lr * grad

        if last_loss is not None and abs(last_loss - loss) < early_stopping_threshold:
            loss_increase_count += 1
        else:
            loss_increase_count = 0

        if loss_increase_count > early_stopping_patience:
            print("Early stopping...")
            break

        last_loss = loss

        print(f"Epoch {epoch}, Loss: {loss}, Output: {output}, Learning Rate: {adaptive_lr}")

    return theta_value

#Test 
init_theta = np.pi / 71
optimized_theta = quantum_perceptron_training(init_theta)
print(f"Initial Theta: {init_theta} \n Optimized Theta: {optimized_theta}")
