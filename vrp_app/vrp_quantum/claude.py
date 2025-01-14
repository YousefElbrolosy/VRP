import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple
import networkx as nx

class VRPQAOA:
    def __init__(self, n_locations: int, n_vehicles: int, distance_matrix: np.ndarray):
        """
        Initialize VRP QAOA solver
        
        Args:
            n_locations: Number of locations (including depot)
            n_vehicles: Number of vehicles
            distance_matrix: Matrix of distances between locations
        """
        self.n_locations = n_locations
        self.n_vehicles = n_vehicles
        self.distance_matrix = distance_matrix
        
        # Number of qubits needed = n_locations * (n_locations-1)
        self.n_qubits = n_locations * (n_locations - 1)
        
        # Initialize the device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
    def _get_ising_hamiltonian(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert VRP to Ising formulation
        
        Returns:
            J: Coupling matrix
            h: Local field vector
            c: Constant offset
        """
        # Initialize matrices
        Q = np.zeros((self.n_qubits, self.n_qubits))
        g = np.zeros(self.n_qubits)
        
        # Penalty coefficient for constraints
        A = np.max(self.distance_matrix) * 10
        
        # Create helper matrices for constraints
        for i in range(self.n_locations):
            # Source constraints
            source_indices = [j for j in range(self.n_locations) if j != i]
            for idx1 in source_indices:
                for idx2 in source_indices:
                    if idx1 != idx2:
                        Q[i*(self.n_locations-1) + idx1][i*(self.n_locations-1) + idx2] += A
                
            # Target constraints
            target_indices = [j for j in range(self.n_locations) if j != i]
            for idx1 in target_indices:
                for idx2 in target_indices:
                    if idx1 != idx2:
                        Q[idx1*(self.n_locations-1) + i][idx2*(self.n_locations-1) + i] += A
                        
        # Add distance terms
        for i in range(self.n_locations):
            for j in range(self.n_locations):
                if i != j:
                    idx = i*(self.n_locations-1) + (j if j < i else j-1)
                    g[idx] += self.distance_matrix[i][j]
        
        # Vehicle constraints for depot
        depot_source_indices = range(self.n_locations-1)
        depot_target_indices = range(self.n_locations-1)
        
        for idx1 in depot_source_indices:
            for idx2 in depot_source_indices:
                if idx1 != idx2:
                    Q[idx1][idx2] += A
                    
        for idx1 in depot_target_indices:
            for idx2 in depot_target_indices:
                if idx1 != idx2:
                    Q[idx1*(self.n_locations-1)][idx2*(self.n_locations-1)] += A
        
        # Convert QUBO to Ising
        J = -Q/4
        h = g/2 + np.sum(Q/4, axis=1)
        c = np.sum(g/2) + np.sum(Q/4)
        
        return J, h, c
        
    def cost_hamiltonian(self, params, wires):
        """
        Implement the cost Hamiltonian
        """
        J, h, _ = self._get_ising_hamiltonian()
        
        # Add ZZ interactions
        for i in range(self.n_qubits):
            for j in range(i+1, self.n_qubits):
                if abs(J[i,j]) > 1e-10:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * J[i,j] * params, wires=j)
                    qml.CNOT(wires=[i, j])
        
        # Add Z terms
        for i in range(self.n_qubits):
            if abs(h[i]) > 1e-10:
                qml.RZ(2 * h[i] * params, wires=i)

    def mixer_hamiltonian(self, params, wires):
        """
        Implement the mixer Hamiltonian
        """
        for wire in wires:
            qml.RX(2 * params, wires=wire)

    def circuit(self, params, steps):
        """
        Implement QAOA circuit
        
        Args:
            params: List of gamma and beta parameters
            steps: Number of QAOA steps (p)
            
        Returns:
            Expectation value of the cost Hamiltonian
        """
        @qml.qnode(self.dev)
        def qaoa_circuit(params, steps):
            # Initialize in superposition
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
                
            # Implement QAOA steps
            for i in range(steps):
                self.cost_hamiltonian(params[i], range(self.n_qubits))
                self.mixer_hamiltonian(params[steps + i], range(self.n_qubits))
                
            return qml.expval(qml.PauliZ(0))  # Measure first qubit
        
        return qaoa_circuit(params, steps)
    
    def optimize(self, steps: int, n_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Optimize the QAOA parameters
        
        Args:
            steps: Number of QAOA steps (p)
            n_iterations: Number of optimization iterations
            
        Returns:
            optimal_params: Optimal gamma and beta parameters
            min_cost: Minimum cost found
        """
        # Initialize parameters
        init_params = np.random.uniform(0, 2*np.pi, 2*steps)
        
        # Define cost function
        def cost(params):
            return self.circuit(params, steps)
        
        # Optimize parameters
        opt = qml.GradientDescentOptimizer(stepsize=0.1)
        params = init_params
        
        for i in range(n_iterations):
            params = opt.step(cost, params)
            
        return params, cost(params)

    def get_solution(self, optimal_params: np.ndarray, steps: int) -> List[List[int]]:
        """
        Get the VRP solution from optimal parameters
        
        Args:
            optimal_params: Optimal QAOA parameters
            steps: Number of QAOA steps used
            
        Returns:
            routes: List of vehicle routes
        """
        # Run circuit with optimal parameters and measure all qubits
        @qml.qnode(self.dev)
        def measure_circuit(params, steps):
            # Initialize in superposition
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
                
            # Implement QAOA steps
            for i in range(steps):
                self.cost_hamiltonian(params[i], range(self.n_qubits))
                self.mixer_hamiltonian(params[steps + i], range(self.n_qubits))
                
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        measurements = measure_circuit(optimal_params, steps)
        
        # Convert measurements to binary
        binary_solution = [1 if m > 0 else 0 for m in measurements]
        
        # Convert binary solution to routes
        routes = []
        current_route = [0]  # Start at depot
        
        # Extract routes from binary solution
        for i in range(self.n_vehicles):
            while len(current_route) < self.n_locations:
                current = current_route[-1]
                next_idx = -1
                
                # Find next location in route
                for j in range(self.n_locations):
                    if j not in current_route:
                        idx = current*(self.n_locations-1) + (j if j < current else j-1)
                        if binary_solution[idx] == 1:
                            next_idx = j
                            break
                            
                if next_idx == -1:
                    break
                    
                current_route.append(next_idx)
                
            if len(current_route) > 1:
                routes.append(current_route)
                current_route = [0]  # Start new route at depot
                
        return routes

# Example usage:
if __name__ == "__main__":
    # Example problem instance (4,2) from the paper
    n_locations = 4
    n_vehicles = 2
    distances = np.array([
        [0.0, 36.84, 5.06, 30.63],
        [36.84, 0.0, 24.55, 63.22],
        [5.06, 24.55, 0.0, 15.50],
        [30.63, 63.22, 15.50, 0.0]
    ])
    
    # Create and run VRP QAOA solver
    vrp_solver = VRPQAOA(n_locations, n_vehicles, distances)
    optimal_params, min_cost = vrp_solver.optimize(steps=12)
    routes = vrp_solver.get_solution(optimal_params, steps=12)
    
    print(f"Optimal routes: {routes}")
    print(f"Minimum cost: {min_cost}")