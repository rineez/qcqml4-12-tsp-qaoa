import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliSumOp, PauliOp
from qiskit.quantum_info import Pauli
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

def create_tsp_graph():
    """
    Create a simple TSP problem with 3 cities labeled A, B, C.
    Returns:
        - distances: Distance matrix
        - cities: List of city labels
    """
    # Distance matrix for 3 cities (symmetric)
    distances = np.array([
        [0, 2, 3],  # Distances from A to [A, B, C]
        [2, 0, 1],  # Distances from B to [A, B, C]
        [3, 1, 0]   # Distances from C to [A, B, C]
    ])
    cities = ['A', 'B', 'C']
    return distances, cities

def visualize_graph(distances, cities, path=None):
    """
    Visualize the TSP graph with optional path highlighting.
    Args:
        distances: Distance matrix
        cities: List of city labels
        path: Optional path to highlight (list of city indices)
    """
    G = nx.Graph()
    
    # Add edges with weights
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            G.add_edge(cities[i], cities[j], weight=distances[i][j])
    
    # Create layout
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(8, 6))
    
    # Draw the basic graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500)
    nx.draw_networkx_labels(G, pos)
    
    # Draw edges with weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    if path is None:
        # Draw all edges if no path is specified
        nx.draw_networkx_edges(G, pos)
        plt.title("Initial TSP Graph")
    else:
        # Draw the optimal path
        path_edges = list(zip(path[:-1], path))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                             edge_color='r', width=2)
        plt.title("TSP Solution Path")
    
    plt.axis('off')
    plt.show()

def create_cost_hamiltonian(distances):
    """
    Create the cost Hamiltonian for the QAOA.
    The cost Hamiltonian encodes the distances between cities.
    Each term represents the cost of traveling between two cities.
    """
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
    """
    Create the constraint Hamiltonian for the QAOA.
    This ensures:
    1. Only one city is visited at each time step
    2. Each city is visited exactly once
    """
    n_qubits = n_cities * n_cities
    constraint_ops = []
    
    # Constraint 1: One city per time step
    for t in range(n_cities):
        for i, j in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t
            qubit2 = j * n_cities + t
            
            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = Pauli(''.join(pauli_str))
            
            constraint_ops.append(PauliOp(pauli, 1.0))
    
    # Constraint 2: Each city visited exactly once
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

def decode_solution(result, n_cities):
    """
    Decode the QAOA result into a valid TSP path.
    Returns the path as a list of city indices.
    """
    # Get the most probable state
    counts = result.eigenstate
    max_prob_state = np.argmax(np.abs(counts))
    
    # Convert to binary string
    binary = format(max_prob_state, f'0{n_cities * n_cities}b')
    
    # Reshape into time steps x cities matrix
    state_matrix = np.array(list(binary)).reshape(n_cities, n_cities)
    
    # Find the city visited at each time step
    path = []
    for t in range(n_cities):
        city = np.argmax(state_matrix[:, t])
        path.append(city)
    
    # Add the first city again to complete the cycle
    path.append(path[0])
    
    return path

def solve_tsp_with_qaoa():
    """
    Main function to solve the TSP using QAOA.
    Returns the optimal path and its visualization.
    """
    # Create problem instance
    distances, cities = create_tsp_graph()
    n_cities = len(distances)
    
    # Visualize initial graph
    print("Initial TSP Graph:")
    visualize_graph(distances, cities)
    
    # Create Hamiltonians
    cost_hamiltonian = create_cost_hamiltonian(distances)
    constraint_hamiltonian = create_constraint_hamiltonian(n_cities)
    
    # Combine Hamiltonians with a penalty weight for constraints
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
    
    # Decode the solution
    path_indices = decode_solution(result, n_cities)
    path = [cities[i] for i in path_indices]
    
    # Calculate total distance
    total_distance = sum(distances[path_indices[i]][path_indices[i+1]] 
                        for i in range(len(path_indices)-1))
    
    # Print results
    print(f"\nQAOA Results:")
    print(f"Optimal path: {' -> '.join(path)}")
    print(f"Total distance: {total_distance}")
    
    # Visualize solution
    print("\nOptimal Path Visualization:")
    visualize_graph(distances, cities, path)
    
    return path, total_distance

# Execute the solution
if __name__ == "__main__":
    optimal_path, total_distance = solve_tsp_with_qaoa()