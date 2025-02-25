import numpy as np
import networkx as nx

class IzhikevichNetworkSimulation:
    def __init__(self, 
                 N=40, 
                 dt=0.1, 
                 tmax=1000,
                 graph_type="erdos-renyi",
                 p=0.6,
                 m=1,
                 k=6,
                 grid_dims=(5, 4),
                 a=0.02,
                 b=0.2,
                 c=-50,
                 d=2):
        # Simulation parameters
        self.N = N
        self.dt = dt
        self.tmax = tmax
        self.steps = int(tmax / dt)
        
        # Izhikevich model parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # Graph/network parameters
        self.graph_type = graph_type
        self.p = p
        self.m = m
        self.k = k
        self.grid_dims = grid_dims
        
        # Create network and store connectivity metadata
        self.G = self.create_graph(N, graph_type, p, m, k, grid_dims)
        self.metadata = {}
        for node in self.G.nodes():
            self.metadata[node] = {
                "neighbors": list(self.G.neighbors(node)),
                "degree": self.G.degree(node)
            }
        
        # Build adjacency matrix for gap junctions
        self.adj_matrix = np.zeros((N, N))
        for (u, v) in self.G.edges():
            g_gap = np.random.uniform(0.1, 0.3)
            self.adj_matrix[u, v] = g_gap
            self.adj_matrix[v, u] = g_gap
        
        # External input: stimulate 3 random neurons
        self.I_ext = np.zeros(N)
        stimulated_neurons = np.random.choice(N, size=3, replace=False)
        self.I_ext[stimulated_neurons] = 20.0
        
        # Initialize state variables:
        # V_history: voltage history for each neuron (time x neuron)
        self.V_history = np.full((self.steps, N), -65.0)
        # U_history: recovery variable history; initial U is b*V[0]
        self.U_history = np.zeros((self.steps, N))
        self.U_history[0, :] = b * self.V_history[0]

    @staticmethod
    def create_graph(N, graph_type="erdos-renyi", p=0.15, m=2, k=4, grid_dims=None):
        """Create a networkx graph given the type and parameters."""
        if graph_type == "erdos-renyi":
            G = nx.erdos_renyi_graph(N, p)
        elif graph_type == "powerlaw":
            G = nx.barabasi_albert_graph(N, m)
        elif graph_type == "ladder":
            G = nx.ladder_graph(N // 2)
        elif graph_type == "star":
            G = nx.star_graph(N - 1)
        elif graph_type == "watts-strogatz":
            G = nx.watts_strogatz_graph(N, k, p)
        elif graph_type == "complete":
            G = nx.complete_graph(N)
        elif graph_type == "cycle":
            G = nx.cycle_graph(N)
        elif graph_type == "path":
            G = nx.path_graph(N)
        elif graph_type == "grid2d":
            if grid_dims is None:
                side = int(np.sqrt(N))
                G = nx.grid_2d_graph(side, side)
            else:
                rows, cols = grid_dims
                G = nx.grid_2d_graph(rows, cols)
            G = nx.convert_node_labels_to_integers(G)
        else:
            raise ValueError(f"Unrecognized graph type: {graph_type}")
        return G

    def run_simulation(self):
        """Run the simulation and record V and U histories for every neuron."""
        for t in range(self.steps - 1):
            # Copy current U values for simultaneous updates
            U_current = self.U_history[t, :].copy()
            for i in range(self.N):
                # Compute gap junction current from neighboring neurons
                I_gap = np.sum(self.adj_matrix[i, :] * (self.V_history[t, :] - self.V_history[t, i]))
                I_net = self.I_ext[i] + I_gap

                # Calculate derivatives using the Izhikevich model equations
                dVdt = 0.04 * self.V_history[t, i]**2 + 5 * self.V_history[t, i] + 140 - U_current[i] + I_net
                dUdt = self.a * (self.b * self.V_history[t, i] - U_current[i])
                
                # Update voltage for next time step
                self.V_history[t + 1, i] = self.V_history[t, i] + dVdt * self.dt
                # Update recovery variable in temporary storage
                U_current[i] += dUdt * self.dt

                # Spike reset: if voltage reaches threshold, reset V and adjust U
                if self.V_history[t + 1, i] >= 30:
                    self.V_history[t + 1, i] = self.c
                    U_current[i] += self.d
            self.U_history[t + 1, :] = U_current.copy()

# Example usage:
if __name__ == "__main__":
    sim = IzhikevichNetworkSimulation()
    sim.run_simulation()
    
    # Access the full voltage history for each neuron:
    V_history = sim.V_history  # shape (steps, N)
    
    # Access the full recovery variable (U) history for each neuron:
    U_history = sim.U_history  # shape (steps, N)
    
    # Access connectivity metadata for each neuron:
    connectivity_metadata = sim.metadata
    
    # Now V_history, U_history, and connectivity_metadata contain the requested numbers.
    print("Voltage history shape:", V_history.shape)
    print("Recovery variable history shape:", U_history.shape)
    print("Sample neuron metadata:", connectivity_metadata[0])
