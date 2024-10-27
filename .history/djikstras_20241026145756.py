import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations

class TSPSolver:
    def __init__(self, distances, cities):
        """
        Initialize the TSP solver with distance matrix and city labels.
        
        Args:
            distances (np.array): Matrix of distances between cities
            cities (list): List of city labels
        """
        self.distances = distances
        self.cities = cities
        self.n_cities = len(cities)
        self.graph = self._create_graph()

    def _create_graph(self):
        """
        Create a NetworkX graph from the distance matrix.
        """
        G = nx.Graph()
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                G.add_edge(self.cities[i], self.cities[j], 
                          weight=self.distances[i][j])
        return G

    def visualize_graph(self, path=None):
        """
        Visualize the graph with optional path highlighting.
        
        Args:
            path (list): Optional path to highlight
        """
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=500)
        nx.draw_networkx_labels(self.graph, pos)
        
        # Draw edges
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
        
        if path is None:
            # Draw all edges if no path is specified
            nx.draw_networkx_edges(self.graph, pos)
            plt.title("Initial TSP Graph")
        else:
            # Draw the optimal path
            path_edges = list(zip(path[:-1], path))
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, 
                                 edge_color='r', width=2)
            plt.title("TSP Solution Path (Dijkstra's)")
        
        plt.axis('off')
        plt.show()

    def dijkstra_shortest_path(self, start, end):
        """
        Find shortest path between two cities using Dijkstra's algorithm.
        
        Args:
            start (str): Starting city
            end (str): Ending city
            
        Returns:
            tuple: (path, distance)
        """
        path = nx.dijkstra_path(self.graph, start, end, weight='weight')
        distance = nx.dijkstra_path_length(self.graph, start, end, weight='weight')
        return path, distance

    def solve_tsp_with_dijkstra(self):
        """
        Solve TSP by trying all possible permutations and using Dijkstra's algorithm
        for path finding between cities.
        
        Returns:
            tuple: (best_path, shortest_distance)
        """
        # Try all possible city orderings (except starting city which is fixed)
        start_city = self.cities[0]
        other_cities = self.cities[1:]
        
        best_path = None
        shortest_distance = float('inf')
        
        # Try each permutation of cities (excluding start city)
        for perm in permutations(other_cities):
            current_path = [start_city]
            current_distance = 0
            current_city = start_city
            
            # Go through each city in the permutation
            for next_city in perm:
                # Find shortest path to next city using Dijkstra's
                path_segment, segment_distance = self.dijkstra_shortest_path(
                    current_city, next_city)
                
                # Add path segment (excluding first city as it's already included)
                current_path.extend(path_segment[1:])
                current_distance += segment_distance
                current_city = next_city
            
            # Return to start city
            final_segment, final_distance = self.dijkstra_shortest_path(
                current_city, start_city)
            current_path.extend(final_segment[1:])
            current_distance += final_distance
            
            # Update best path if current path is shorter
            if current_distance < shortest_distance:
                shortest_distance = current_distance
                best_path = current_path
        
        return best_path, shortest_distance

def main():
    """
    Main function to demonstrate TSP solution using Dijkstra's algorithm.
    """
    # Create sample problem
    distances = np.array([
        [0, 2, 4],  # Distances from A to [A, B, C]
        [2, 0, 1],  # Distances from B to [A, B, C]
        [4, 1, 0]   # Distances from C to [A, B, C]
    ])
    cities = ['A', 'B', 'C']
    
    # Create solver instance
    solver = TSPSolver(distances, cities)
    
    # Show initial graph
    print("Initial Graph:")
    solver.visualize_graph()
    
    # Solve TSP
    print("\nSolving TSP using Dijkstra's algorithm...")
    best_path, shortest_distance = solver.solve_tsp_with_dijkstra()
    
    # Print results
    print(f"\nResults:")
    print(f"Optimal path: {' -> '.join(best_path)}")
    print(f"Total distance: {shortest_distance}")
    
    # Visualize solution
    print("\nOptimal Path Visualization:")
    solver.visualize_graph(best_path)

if __name__ == "__main__":
    main()