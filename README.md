# Quantum Support Vector Machine (QSVM) with Qiskit

## Introduction

This Colab notebook demonstrates the implementation of a Quantum Support Vector Machine (QSVM) using Qiskit, a quantum computing framework in Python. The QSVM is a variant of classical Support Vector Machine (SVM) that utilizes quantum computing resources for training and prediction tasks. This approach is particularly valuable when classical SVM struggles to find an effective decision boundary due to limitations in classical computing resources.

## Quantum Feature Maps

To start, we'll explore the concept of Quantum Feature Maps, which are circuits designed to map input data points into higher-dimensional spaces. This is achieved using the ZZFeatureMap provided by Qiskit. The following code illustrates the creation and visualization of a ZZFeatureMap circuit:

```python
!pip install pylatexenc
from qiskit.circuit.library import ZZFeatureMap

num_qubits = 4
x = np.random.random(num_qubits)
data = ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement="linear")
data.assign_parameters(x, inplace=True)
data.decompose().draw("mpl", style="iqx", scale=1.4)
```
Building a Quantum Kernel

Next, we build a quantum kernel using Qiskit. The following code demonstrates the creation of a ZZFeatureMap circuit with increased feature dimension:

python

from qiskit import BasicAer, transpile, QuantumCircuit

backend = BasicAer.get_backend("qasm_simulator")
shots = 1024

dimension = 2
feature_map = ZZFeatureMap(dimension, reps=1)

This code prepares a quantum circuit that applies a ZZFeatureMap to a set of qubits and then simulates the circuit using a QASM simulator backend.

We also define an evaluate_kernel function to calculate the dot product of two input vectors in a higher-dimensional space.

```python

def evaluate_kernel(x_i, x_j):
    # ... (refer to code snippet for implementation details)
    return counts.get("0" * dimension, 0) / shots
```
Using Qiskit Nature

To leverage Qiskit Nature for quantum machine learning tasks, we install the necessary packages:

```python

!pip install qiskit-aer
!pip install qiskit-algorithms
!pip install qiskit-machine-learning
```
We then utilize Qiskit Nature to create a FidelityQuantumKernel and evaluate the kernel for two data points:

```python

# ... (refer to code snippet for complete implementation)
kernel_value = kernel.evaluate(X[2], X[3])
```
Classification

Finally, we apply the QSVM for classification using scikit-learn. We generate a set of points forming a circle and train the QSVM:

```python

# ... (refer to code snippet for complete implementation)
qsvm = SVC(kernel=kernel.evaluate)
qsvm.fit(points, labels)
predicted = qsvm.predict(points)
```
The code includes visualization to display correctly and misclassified points.

Feel free to execute this Colab notebook for a hands-on experience with Quantum Support Vector Machines using Qiskit.
