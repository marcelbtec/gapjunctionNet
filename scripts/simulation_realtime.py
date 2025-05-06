import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter  # Add this import for video writing

plt.style.use('dark_background')

def create_graph(N, graph_type="erdos-renyi", p=0.15, m=2, k=4, grid_dims=None):
    """
    Create a networkx graph given the type of graph and relevant parameters.
    """
    if graph_type == "erdos-renyi":
        G = nx.erdos_renyi_graph(N, p)
    elif graph_type == "powerlaw":
        # Barabási–Albert, scale-free
        G = nx.barabasi_albert_graph(N, m)
    elif graph_type == "ladder":
        # Ladder graph requires an even number of nodes to make perfect pairs
        G = nx.ladder_graph(N // 2)
    elif graph_type == "star":
        # star_graph(n) has n+1 nodes, so star_graph(N-1) -> N total
        G = nx.star_graph(N - 1)
    elif graph_type == "watts-strogatz":
        # k is the number of nearest neighbors, p is the probability of rewiring
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

# -------------------
# Simulation parameters
# -------------------
dt = 0.5
tmax = 600
steps = int(tmax / dt)
N = 20

# -------------------
# Izhikevich model parameters
# -------------------
a = 0.02
b = 0.2
c = -50
d = 2

# -------------------
# Choose network type and parameters
# -------------------
graph_type = "erdos-renyi"  # Options: erdos-renyi, powerlaw, ladder, star, watts-strogatz, complete, cycle, path, grid2d
p = 0.3   # Rewiring probability (also used in erdos-renyi)
m = 1          # For Barabási–Albert
k = 6          # For Watts-Strogatz
grid_dims = (5, 4)  # Only if graph_type="grid2d"

# Create the graph
G = create_graph(
    N=N,
    graph_type=graph_type,
    p=p,
    m=m,
    k=k,
    grid_dims=grid_dims
)

# -------------------
# Build adjacency matrix for gap junctions
# -------------------
adj_matrix = np.zeros((N, N))
for (u, v) in G.edges():
    g_gap = np.random.uniform(0.1, 0.3)
    adj_matrix[u, v] = g_gap
    adj_matrix[v, u] = g_gap

# -------------------
# External input
# -------------------
I_ext = np.zeros(N)
stimulated_neurons = np.random.choice(N, size=3, replace=False)
I_ext[stimulated_neurons] = 20.0

# -------------------
# State variables
# -------------------
V = np.full((steps, N), -65.0)
U = b * V[0]

# -------------------
# Simulation loop
# -------------------
for t in range(steps - 1):
    for i in range(N):
        I_gap = np.sum(adj_matrix[i, :] * (V[t, :] - V[t, i]))
        I_net = I_ext[i] + I_gap

        dVdt = 0.04 * V[t, i]**2 + 5 * V[t, i] + 140 - U[i] + I_net
        dUdt = a * (b * V[t, i] - U[i])

        V[t + 1, i] = V[t, i] + dVdt * dt
        U[i]       += dUdt * dt

        # Spike reset
        if V[t + 1, i] >= 30:
            V[t + 1, i] = c
            U[i]       += d

# -------------------
# Prepare data for plotting
# -------------------
avg_V = np.mean(V, axis=1)
time_array = np.arange(steps) * dt

# Reconstruct U history to obtain the network's overall state trajectory.
U_hist = np.empty_like(V)
U_hist[0, :] = b * V[0, :]
for t in range(steps - 1):
    U_next = U_hist[t, :] + a * (b * V[t, :] - U_hist[t, :]) * dt
    spike_indices = (V[t+1, :] == c)
    U_next[spike_indices] += d
    U_hist[t+1, :] = U_next
avg_U = np.mean(U_hist, axis=1)

# -------------------
# Visualization using GridSpec for three panels
# -------------------
fig = plt.figure(figsize=(18, 10))
# Three rows: top row for network/time series, second row for neuron activations, third row for state space plot.
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

# Top row: two columns for network and time series plots
gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
ax_net = fig.add_subplot(gs_top[0])
ax_ts = fig.add_subplot(gs_top[1])

# Second row: grid of individual neuron activation subplots.
num_cols = 6
num_rows = int(np.ceil(N / num_cols))
gs_bottom = gridspec.GridSpecFromSubplotSpec(num_rows, num_cols, subplot_spec=gs[1])
activation_axes = []
neuron_lines = []   # Each neuron's animated line
vertical_lines = [] # Vertical marker for current time
for i, spec in enumerate(gs_bottom):
    ax = fig.add_subplot(spec)
    if i < N:
        activation_axes.append(ax)
        line_obj, = ax.plot([], [], color='blue', lw=1)
        neuron_lines.append(line_obj)
        vline = ax.axvline(x=0, color='red', lw=1)
        vertical_lines.append(vline)
        ax.set_title(f"Neuron {i}", fontsize=8, loc='center', pad=6)
        ax.set_xlim(0, tmax)
        ax.set_ylim(V.min(), V.max())
        ax.tick_params(axis='both', labelsize=6)
    else:
        ax.axis('off')

# Third row: split into two columns for state space (left) and raster plot (right)
gs_bottomrow = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], width_ratios=[1, 1])
ax_state = fig.add_subplot(gs_bottomrow[0])
ax_raster = fig.add_subplot(gs_bottomrow[1])

# State space plot (left, square)
ax_state.set_title("State Space: Average V vs. Average U")
ax_state.set_xlabel("Average V (mV)")
ax_state.set_ylabel("Average U")
ax_state.set_xlim(np.min(avg_V) - 10, np.max(avg_V) + 10)
ax_state.set_ylim(np.min(avg_U) - 5, np.max(avg_U) + 5)
line_state, = ax_state.plot([], [], color='green', lw=2)

# Raster plot (right)
ax_raster.set_title("Spike Raster Plot")
ax_raster.set_xlabel("Time (ms)")
ax_raster.set_ylabel("Neuron Index")
ax_raster.set_xlim(0, tmax)
ax_raster.set_ylim(-0.5, N-0.5)

# Compute spike times for all neurons (V crosses 30 mV)
spike_times = [np.where(V[1:, i] == c)[0] * dt for i in range(N)]
spike_dots = []
for i, times in enumerate(spike_times):
    dot, = ax_raster.plot([], [], ls='', marker='|', color='white', markersize=8)
    spike_dots.append(dot)

# --- Network Plot (Left Top Panel) ---
pos = nx.spring_layout(G)  # fixed layout for reproducibility
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_color=V[0],
    cmap="coolwarm",
    node_size=500,
    vmin=-80, vmax=40,
    ax=ax_net
)
nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax_net, edge_color='#CCCCCC')
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax_net)
cbar = plt.colorbar(nodes, ax=ax_net, label="Membrane Voltage (mV)")
ax_net.set_title(f"{graph_type.capitalize()} network\n t=0.0 ms")

# --- Time Series Plot (Right Top Panel) ---
ax_ts.set_title("Average Network Voltage Over Time")
ax_ts.set_xlabel("Time (ms)")
ax_ts.set_ylabel("Avg Voltage (mV)")
line_ts, = ax_ts.plot([], [], color='blue', lw=2)
ax_ts.set_xlim(0, tmax)
ax_ts.set_ylim(avg_V.min() - 10, avg_V.max() + 10)

# -------------------
# Animation functions
# -------------------
def init_anim():
    nodes.set_array(V[0])
    line_ts.set_data([], [])
    for neuron_line in neuron_lines:
        neuron_line.set_data([], [])
    for vline in vertical_lines:
        vline.set_xdata([0, 0])
    line_state.set_data([], [])
    for dot in spike_dots:
        dot.set_data([], [])
    return (nodes, line_ts, line_state, *neuron_lines, *vertical_lines, *spike_dots)

def update_anim(frame):
    ax_net.set_title(f"{graph_type.capitalize()} network\n t = {frame * dt:.1f} ms")
    nodes.set_array(V[frame])
    line_ts.set_data(time_array[:frame], avg_V[:frame])
    for i, neuron_line in enumerate(neuron_lines):
        neuron_line.set_data(time_array[:frame], V[:frame, i])
    for vline in vertical_lines:
        vline.set_xdata([frame * dt, frame * dt])
    line_state.set_data(avg_V[:frame], avg_U[:frame])
    # Raster plot update
    for i, dot in enumerate(spike_dots):
        spike_mask = spike_times[i] <= frame * dt
        dot.set_data(spike_times[i][spike_mask], np.full(np.sum(spike_mask), i))
    return (nodes, line_ts, line_state, *neuron_lines, *vertical_lines, *spike_dots)

ani = animation.FuncAnimation(
    fig, update_anim,
    frames=steps,
    interval=50,
    init_func=init_anim,
    blit=False
)

plt.tight_layout()
plt.show()

def save_animation_as_movie(ani, filename='simulation.mp4', fps=20, dpi=300):
    """
    Save the animation as a movie file.
    
    Parameters:
    -----------
    ani : matplotlib.animation.FuncAnimation
        The animation object to save
    filename : str
        Name of the output file (default: 'simulation.mp4')
    fps : int
        Frames per second (default: 20)
    dpi : int
        Dots per inch for the output video (default: 300)
    """
    writer = FFMpegWriter(fps=fps)
    ani.save(filename, writer=writer, dpi=dpi)
    print(f"Animation saved as {filename}")

# Uncomment the following line to save the animation as a movie
#save_animation_as_movie(ani, filename='simulation.mp4', fps=20, dpi=300)

