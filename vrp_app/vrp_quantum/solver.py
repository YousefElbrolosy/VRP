import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

class VRPQAOA:
    def __init__(self, n, k, distance_matrix):
        """
        n: number of nodes
        k: number of vehicles
        distance_matrix: matrix of distances between nodes
        """
        self.n = n
        self.k = k
        self.distance_matrix = distance_matrix
        self.num_of_qubits = n*(n-1)
        self.A = 2
        # Number of layers (precision)
        self.p = 6
        self.dev = qml.device("lightning.qubit", wires=self.num_of_qubits)


    # Preparing the elementary variables    
    
    def get_x_vector(self):
        """
        uses: self.distance_matrix
        size: (n*(n-1)) By 1

        Returns:

        x_vector: vector of binary decision variables representing the edges
        """
        # The total number of possible edges in a directed graph with n nodes is n*(n-1).
        x_vector = np.zeros(self.num_of_qubits)

        # Fill the vector with placeholders for the binary variables
        # Placeholder: Each x_ij corresponds to an index in the vector
        for i in range(self.n):
            for j in range(self.n):
                if (i != j and self.distance_matrix[i][j]!=0):  # Avoid self-loops (i.e., no edges from a node to itself)
                    index = i * (self.n - 1) + j if j < i else i * (self.n - 1) + j - 1
                    x_vector[index] = 1  # Assign a binary variable placeholder

        return x_vector
    

    # this places a 1 even if there is no edge connecting between the nodes
    def get_z_source(self, source_index):
        """
        Args:
            index: int, the index of the source node
        size: (n*(n-1)) By 1
        Returns:

        z_source: vector of binary decision variables representing the source node
        """
        z_source = np.zeros(self.num_of_qubits)

        # Fill the vector with placeholders for the binary variables
        for i in range(self.n):
            for j in range(self.n):
                if (i == source_index and i != j):
                    index = i * (self.n - 1) + j if j < i else i * (self.n - 1) + j - 1
                    z_source[index] = 1  # Assign a binary variable placeholder
        
        return z_source
    
    def get_z_target(self, target_index):
        """
        size: (n*(n-1)) By 1
        Returns:

        z_sink: vector of binary decision variables representing the sink/target node
        """
        z_target = np.zeros(self.num_of_qubits)

        # Fill the vector with placeholders for the binary variables
        for i in range(self.n):
            for j in range(self.n):
                if (j == target_index and i != j):
                    index = i * (self.n - 1) + j if j < i else i * (self.n - 1) + j - 1
                    z_target[index] = 1  # Assign a binary variable placeholder
        
        return z_target
    
    def get_w_vector(self):
        """
        uses: self.distance_matrix
        size: (n*(n-1)) By 1
        Returns:

        w_vector: vector of weights, representing the distance from node i to j
        """

    def get_Q(self):
        """
        Returns:
    
        Q: (the quadratic coefficient) which represents the edge
        weight i.e., coupling or interaction between two nodes.

        """
    def get_g(self):
        """
        Returns:

        g: the linear coefficient g which represents the node weight i.e.,
        contribution from individual nodes

        """
    def get_c(self):
        """
        Returns:

        c: constant offset

        """        


    # Preparing the variables needed for the cost hamiltonian

    def get_J_Matrix(self):
        """
        size: (n*(n-1)) By (n*(n-1))
        Returns:

        J: where J is a matrix to keep track of the interactions
        """

    def get_h_vector(self):
        """
        Returns:

        h: vector of linear coefficients

        """
    def get_d_offset(self):
        """
        Returns:

        d_offset: constant offset

        """


    # Preparing the cost hamiltonian
    def get_cost_hamiltonian(self):
        """
        Returns:

        cost_hamiltonian: the cost hamiltonian

        """

    # Preparing the mixer hamiltonian
    def get_mixer_hamiltonian(self):
        """
        Returns:

        mixer_hamiltonian: the mixer hamiltonian

        """

    # Preparing the QAOA circuit

    # create the qaoa layer
    def qaoa_layer(self, gamma, alpha):
        """
        Args: (basically parameters for classical optimization of the quantum circuit)
            gamma: float, the angle of the rotation in the cost Hamiltonian
            alpha: float, the angle of the rotation in the mixer Hamiltonian
        """
        qaoa.cost_layer(gamma, self.get_cost_hamiltonian())
        qaoa.mixer_layer(alpha, self.get_mixer_hamiltonian())

    # create the qaoa circuit
    def circuit(self, params, **kwargs):
        """
        Args:
            params: list of parameters for the quantum circuit
        """
        for w in self.num_of_qubits:
            qml.Hadamard(wires=w)
        qml.layer(self.qaoa_layer, self.p, params[0], params[1])
    

    def cost_function_circuit(self):
        """
        Returns:
            the cost of the circuit using the current parameters
        """
        @qml.qnode(self.dev)
        def cost_function(params):
            self.circuit(params)
            return qml.expval(self.get_cost_hamiltonian())
        return cost_function
    
    def optimize(self, n_iterations=100):
        """
        Args:
            n_iterations: number of optimization iterations
        Returns:
            params: optimized parameters for the quantum circuit
        """

        optimizer = qml.GradientDescentOptimizer()
        steps = n_iterations
        params = np.array([[0.5]*self.p]*2, requires_grad=True)
        for _ in range(steps):
            params = optimizer.step(self.cost_function_circuit, params)
        return params
    
    def get_probability_distribution(self, params):
        """
        Args:
            params: optimized parameters for the quantum circuit
        Returns:
            probability_distribution: the probability distribution of the solution
        """
        @qml.qnode(self.dev)
        def probability_distribution(params):
            self.circuit(params)
            return qml.probs(wires=self.num_of_qubits)
        return probability_distribution(params)
    
    # used to visualize the solution 
    
    def get_binary_solution(self):
        """
        uses: get_probability_distribution to extract the solution with the highest probability
        Returns:
            binary_solution: the binary solution of the problem
        """
    
    def get_solution_cost(self):
        """
        calculates the solution based on the distance matrix and the binary solution
        Returns:
            cost: the cost of the whole trip
        """

    # preferebly visualizes the solution using a graph of the form
    def translate_binary_solution(self):
        """
        uses the binary format of the solution to output the edges which are taken and which are not
        Returns:
            solution: the solution in a human-readable format e:g [(0, 1), (1, 2), (2, 3), (3, 4)]
        """


if __name__ == "__main__":
    # Distance matrix D1 from the problem
    D1 = np.array([
        [0.00, 36.84, 5.06, 30.63],
        [36.84, 0.00, 24.55, 63.22],
        [5.06, 24.55, 0.00, 15.50],
        [30.63, 63.22, 15.50, 0.00]
    ])
    vrp = VRPQAOA(4, 2, D1)
    # params = vrp.optimize()
    # probability_distribution = vrp.get_probability_distribution(params)
    # plt.bar(range(2 ** len(vrp.num_of_qubits)), probability_distribution)
    # plt.show()  
    # print(probability_distribution)
    print(vrp.get_x_vector())
    print(vrp.get_z_source(0))
    print(vrp.get_z_target(2))



