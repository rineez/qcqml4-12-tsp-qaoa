import sys
import time
import numpy as np
from itertools import combinations
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli
from graph import visualize_graph


def decode_solution(result, n_cities):
    """
    Decode the QAOA result into a valid TSP path.
    Now properly handles dictionary output from QAOA.
    """
    try:
        # Get the state with highest probability from the counts dictionary
        if hasattr(result, 'eigenstate') and isinstance(result.eigenstate, dict):
            # Handle dictionary of counts
            counts = result.eigenstate
            max_bitstring = max(counts.items(), key=lambda x: x[1])[0]
            binary = format(int(max_bitstring, 2), f'0{n_cities * n_cities}b')
        else:
            # Fallback: create a simple sequential path
            print("Warning: Could not decode quantum state, using fallback path")
            path = list(range(n_cities)) + [0]
            return path
            
        # Convert to state matrix
        state_matrix = np.array(list(map(int, binary))).reshape(n_cities, n_cities)
        
        # Build valid path
        path = []
        used_cities = set()
        
        for t in range(n_cities):
            probs = state_matrix[:, t]
            available = [i for i in range(n_cities) if i not in used_cities]
            
            if not available:
                remaining = list(set(range(n_cities)) - set(path))
                city = remaining[0] if remaining else path[0]
            else:
                city = max(available, key=lambda x: probs[x])
            
            path.append(city)
            used_cities.add(city)
        
        path.append(path[0])  # Complete the cycle
        return path
        
    except Exception as e:
        print(f"Warning: Error in solution decoding: {e}")
        print("Using fallback path")
        # Return a simple sequential path as fallback
        return list(range(n_cities)) + [0]


def create_tsp_graph(N):
    """
    Create a TSP problem with N cities labeled from A, B, C, ..., up to the Nth letter.
    Returns a symmetric distance matrix with zero diagonals.
    """
    # Generate random distance matrix
    distances = np.random.randint(1, 10, size=(N, N))  # Random distances between 1 and 9

    # Make the distance matrix symmetric
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)  # Set diagonal to 0, no self-loops

    # Generate city labels as uppercase letters
    cities = [chr(65 + i) for i in range(N)]  # A, B, C, ..., etc.

    return distances, cities


def create_cost_hamiltonian(distances):
    """
    Create the cost Hamiltonian for the QAOA.
    """
    n_cities = len(distances)
    n_qubits = n_cities * n_cities
    cost_ops = []
    
    # Distance terms
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
    
    # Add strong penalty terms
    penalty = 20.0
    
    # One city per time step
    for t in range(n_cities):
        for i, j in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t
            qubit2 = j * n_cities + t
            
            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = Pauli(''.join(pauli_str))
            
            cost_ops.append(PauliOp(pauli, penalty))
    
    # Each city visited once
    for i in range(n_cities):
        for t1, t2 in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t1
            qubit2 = i * n_cities + t2
            
            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = Pauli(''.join(pauli_str))
            
            cost_ops.append(PauliOp(pauli, penalty))
    
    return sum(cost_ops)


def solve_tsp_with_qaoa(distances, cities):
    """
    Solve TSP using QAOA.
    """
    # Create Hamiltonian
    cost_hamiltonian = create_cost_hamiltonian(distances)
    
    # Set up QAOA
    p = 3  # Number of QAOA layers
    optimizer = COBYLA(maxiter=500)
    
    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(
        backend,
        shots=4096,
        seed_simulator=123,
        seed_transpiler=123
    )
    
    qaoa = QAOA(
        optimizer=optimizer,
        quantum_instance=quantum_instance,
        reps=p,
        initial_point=[np.pi/3] * (2*p)
    )
    
    # Run QAOA
    try:
        result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)
        return result, 0
        
    except Exception as e:
        print(f"Error in QAOA execution: {e}")
        return None, 1


def main():
    # Check if N is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <N>")
        print("<N> is the number of cities (an integer)")
        sys.exit(1)

    # Parse N from command-line arguments
    try:
        N = int(sys.argv[1])
        if N < 2:
            raise ValueError("N must be 2 or greater.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create problem instance
    distances, cities = create_tsp_graph(N)
    n_cities = len(distances)

    # print("\nInitial Path Visualization:")
    # visualize_graph(distances, cities)

    result, err = solve_tsp_with_qaoa(distances, cities)
    if err:
        sys.exit(1)

    # Debug information
    print(f"\nQAOA Result Details:")
    print(f"Result type: {type(result)}")
    print(f"Available attributes: {dir(result)}")
    if hasattr(result, 'eigenstate'):
        print(f"Eigenstate type: {type(result.eigenstate)}")
        
    # Decode solution
    path_indices = decode_solution(result, n_cities)
    path = [cities[i] for i in path_indices]

    # Calculate total distance
    total_distance = sum(distances[path_indices[i]][path_indices[i+1]] 
                        for i in range(len(path_indices)-1))
    
    # Print results
    print(f"\nQAOA Results:")
    print(f"Optimal path: {' -> '.join(path)}")
    print(f"Total distance: {total_distance}")
    
    # Show solution
    print("\nOptimal Path Visualization:")
    visualize_graph(distances, cities, path)
    sys.exit(0)


if __name__ == "__main__":
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    
    runtime_duration = end_time - start_time
    print(f"Script ran for: {runtime_duration // 60:.2f} minutes")
