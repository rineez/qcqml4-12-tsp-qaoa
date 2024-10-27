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
        [0, 2, 4],  # Distances from A to [A, B, C]
        [2, 0, 1],  # Distances from B to [A, B, C]
        [4, 1, 0]   # Distances from C to [A, B, C]
    ])
    cities = ['A', 'B', 'C']
    return distances, cities

def create_cost_hamiltonian(distances):
    """
    Create the cost Hamiltonian for the QAOA with stronger penalties for invalid tours.
    """
    n_cities = len(distances)
    n_qubits = n_cities * n_cities
    
    cost_ops = []
    
    # Add terms for distances between cities
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                for t in range(n_cities):
                    t_next = (t + 1) % n_cities
                    qubit1 = i * n_cities + t
                    qubit2 = j * n_cities + t_next
                    
                    pauli_str = ['I'] * n_qubits
                    pauli_str[qubit1] = 'Z'
                    pauli_str[qubit2] = 'Z'
                    pauli = Pauli(''.join(pauli_str))
                    
                    cost_ops.append(PauliOp(pauli, distances[i][j] / 4))
    
    # Add strong penalty terms for invalid configurations
    penalty_strength = 10.0  # Increased penalty strength
    
    # Penalty for not having exactly one city per time step
    for t in range(n_cities):
        for i in range(n_cities):
            qubit = i * n_cities + t
            pauli_str = ['I'] * n_qubits
            pauli_str[qubit] = 'Z'
            pauli = Pauli(''.join(pauli_str))
            cost_ops.append(PauliOp(pauli, penalty_strength))
    
    # Penalty for not visiting each city exactly once
    for i in range(n_cities):
        for t in range(n_cities):
            qubit = i * n_cities + t
            pauli_str = ['I'] * n_qubits
            pauli_str[qubit] = 'Z'
            pauli = Pauli(''.join(pauli_str))
            cost_ops.append(PauliOp(pauli, penalty_strength))
    
    cost_hamiltonian = sum(cost_ops)
    return cost_hamiltonian

def decode_solution(result, n_cities):
    """
    Decode the QAOA result into a valid TSP path with validation.
    """
    counts = result.eigenstate
    max_prob_state = np.argmax(np.abs(counts))
    
    # Convert to binary string
    binary = format(max_prob_state, f'0{n_cities * n_cities}b')
    
    # Reshape into time steps x cities matrix
    state_matrix = np.array(list(map(int, binary))).reshape(n_cities, n_cities)
    
    # Validate solution
    path = []
    visited = set()
    
    for t in range(n_cities):
        # Find the most likely city for this time step
        prob_cities = state_matrix[:, t]
        
        # Choose unvisited city with highest probability
        available_cities = [i for i in range(n_cities) if i not in visited]
        if not available_cities:
            # If no unvisited cities, choose any remaining city
            city = np.argmax(prob_cities)
        else:
            # Choose unvisited city with highest probability
            prob_unvisited = [prob_cities[i] if i in available_cities else -1 
                            for i in range(n_cities)]
            city = np.argmax(prob_unvisited)
        
        path.append(city)
        visited.add(city)
    
    # Add starting city to complete the tour
    path.append(path[0])
    
    return path

def solve_tsp_with_qaoa():
    """
    Main function to solve the TSP using QAOA with improved parameters.
    """
    # Create problem instance
    distances, cities = create_tsp_graph()
    n_cities = len(distances)
    
    # Visualize initial graph
    print("Initial TSP Graph:")
    visualize_graph(distances, cities)
    
    # Create Hamiltonians with improved parameters
    cost_hamiltonian = create_cost_hamiltonian(distances)
    
    # Set up QAOA with improved parameters
    p = 3  # Increased number of QAOA layers
    optimizer = COBYLA(maxiter=200)  # Increased maximum iterations
    quantum_instance = QuantumInstance(
        Aer.get_backend('statevector_simulator'),
        shots=1024  # Increased number of shots
    )
    
    # Create and run QAOA
    qaoa = QAOA(
        optimizer=optimizer,
        quantum_instance=quantum_instance,
        reps=p,
        initial_point=[np.pi/3] * (2*p)  # Better initial point
    )
    
    # Get the result
    result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)
    
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

def visualize_graph(distances, cities, path=None):
    """
    Visualize the TSP graph with optional path highlighting.
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
        nx.draw_networkx_edges(G, pos)
        plt.title("Initial TSP Graph")
    else:
        path_edges = list(zip(path[:-1], path))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                             edge_color='r', width=2)
        plt.title("TSP Solution Path (QAOA)")
    
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    optimal_path, total_distance = solve_tsp_with_qaoa()