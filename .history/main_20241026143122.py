#Compare the accuracy of the results obtained for diff. no. of nodes and discuss the observations.
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliSumOp, PauliOp
from qiskit.quantum_info import Pauli
import networkx as nx
from itertools import combinations

# Create a simple TSP problem with 3 cities
def create_tsp_graph():
    # Distance matrix for 3 cities (symmetric)
    distances = np.array([
        [0, 2, 3],
        [2, 0, 1],
        [3, 1, 0]
    ])
    return distances

def create_cost_hamiltonian(distances):
    n_cities = len(distances)
    n_qubits = n_cities * n_cities
    
    # Initialize cost operators list
    cost_ops = []
    
    # Add terms for distances between cities
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                for t in range(n_cities):
                    t_next = (t + 1) % n_cities
                    # Add distance cost between consecutive cities in the path
                    qubit1 = i * n_cities + t
                    qubit2 = j * n_cities + t_next
                    
                    # Create Pauli string for the interaction
                    pauli_str = ['I'] * n_qubits
                    pauli_str[qubit1] = 'Z'
                    pauli_str[qubit2] = 'Z'
                    pauli = Pauli(''.join(pauli_str))
                    
                    cost_ops.append(PauliOp(pauli, distances[i][j] / 4))
    
    # Combine all operators
    cost_hamiltonian = sum(cost_ops)
    return cost_hamiltonian

def create_constraint_hamiltonian(n_cities):
    n_qubits = n_cities * n_cities
    constraint_ops = []
    
    # One city per time step
    for t in range(n_cities):
        for i, j in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t
            qubit2 = j * n_cities + t
            
            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = Pauli(''.join(pauli_str))
            
            constraint_ops.append(PauliOp(pauli, 1.0))
    
    # Each city visited exactly once
    for i in range(n_cities):
        for t1, t2 in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t1
            qubit2 = i * n_cities + t2
            
            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = Pauli(''.join(pauli_str))
            
            constraint_ops.append(PauliOp(pauli, 1.0))
    
    constraint_hamiltonian = sum(constraint_ops)
    return constraint_hamiltonian

# Main execution
def solve_tsp_with_qaoa():
    # Create problem instance
    distances = create_tsp_graph()
    n_cities = len(distances)
    
    # Create Hamiltonians
    cost_hamiltonian = create_cost_hamiltonian(distances)
    constraint_hamiltonian = create_constraint_hamiltonian(n_cities)
    
    # Combine Hamiltonians
    total_hamiltonian = cost_hamiltonian + 10.0 * constraint_hamiltonian
    
    # Set up QAOA
    p = 2  # Number of QAOA layers
    optimizer = COBYLA(maxiter=100)
    quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
    
    # Create and run QAOA
    qaoa = QAOA(optimizer=optimizer,
                quantum_instance=quantum_instance,
                reps=p)
    
    # Get the result
    result = qaoa.compute_minimum_eigenvalue(total_hamiltonian)
    
    return result

# Execute the solution
result = solve_tsp_with_qaoa()
print(f"QAOA Results:")
print(f"Minimum eigenvalue found: {result.eigenvalue.real}")
print(f"Optimal parameters: {result.optimal_parameters}")