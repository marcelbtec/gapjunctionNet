import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ---------------------------
# Original code blocks unchanged in logic
# (except we separate steps into smaller chunks)
# ---------------------------

def create_graph(N, graph_type="erdos-renyi", p=0.15, m=2, k=4, grid_dims=None):
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


def initialize_simulation(N=24, dt=0.4, tmax=500,
                         graph_type="path", p=0.2, m=3, k=6, grid_dims=(5,5)):
    """
    Initializes everything needed for the simulation:
    - Graph and adjacency matrix
    - State variables V and U
    - External current
    - We'll store in session_state so we can incrementally update.
    """
    steps = int(tmax / dt)

    # Store basic parameters in session_state
    st.session_state.dt = dt
    st.session_state.tmax = tmax
    st.session_state.steps = steps
    st.session_state.N = N

    st.session_state.a = 0.02
    st.session_state.b = 0.2
    st.session_state.c = -50
    st.session_state.d = 2

    # Create the graph
    G = create_graph(N=N, graph_type=graph_type, p=p, m=m, k=k, grid_dims=grid_dims)
    st.session_state.G = G

    # Build adjacency
    adj_matrix = np.zeros((N, N))
    for (u, v) in G.edges():
        g_gap = np.random.uniform(0.1, 0.3)
        adj_matrix[u, v] = g_gap
        adj_matrix[v, u] = g_gap
    st.session_state.adj_matrix = adj_matrix

    # External input
    I_ext = np.zeros(N)
    stimulated = np.random.choice(N, size=3, replace=False)
    I_ext[stimulated] = 20.0
    st.session_state.I_ext = I_ext

    # State variables
    V = np.full((steps, N), -65.0)
    U = st.session_state.b * V[0]
    st.session_state.V = V
    st.session_state.U = U

    # We'll keep track of current step in simulation
    st.session_state.current_step = 0

    # For convenience, we store results for average voltage, but fill it as we go
    st.session_state.avg_V = np.zeros(steps)
    st.session_state.time_array = np.arange(steps) * dt


def simulation_step():
    """
    Runs exactly ONE iteration (time step) of the simulation
    and updates the data in st.session_state.
    """
    i_step = st.session_state.current_step
    steps = st.session_state.steps
    if i_step >= steps - 1:
        return  # no more steps left

    V = st.session_state.V
    U = st.session_state.U
    dt = st.session_state.dt
    N = st.session_state.N

    a = st.session_state.a
    b = st.session_state.b
    c = st.session_state.c
    d = st.session_state.d

    I_ext = st.session_state.I_ext
    adj_matrix = st.session_state.adj_matrix

    t = i_step
    for i in range(N):
        I_gap = np.sum(adj_matrix[i, :] * (V[t, :] - V[t, i]))
        I_net = I_ext[i] + I_gap

        dVdt = 0.04 * V[t, i]**2 + 5 * V[t, i] + 140 - U[i] + I_net
        dUdt = a * (b * V[t, i] - U[i])

        V[t + 1, i] = V[t, i] + dVdt * dt
        U[i]       += dUdt * dt

        if V[t + 1, i] >= 30:
            V[t + 1, i] = c
            U[i]       += d

    # Compute average voltage at step t
    st.session_state.avg_V[t] = np.mean(V[t, :])

    st.session_state.current_step += 1  # move to next time step


def plot_current_state():
    """
    Plots the current state: 
      - The network with node colors = V at the current step
      - The average voltage up to the current step 
      - And the subplots with each neuron's voltage up to the current step
    Returns the matplotlib figure.
    """

    i_step = st.session_state.current_step
    V = st.session_state.V
    avg_V = st.session_state.avg_V
    time_array = st.session_state.time_array
    steps = st.session_state.steps
    dt = st.session_state.dt
    N = st.session_state.N
    G = st.session_state.G

    # If we haven't started stepping yet, i_step=0
    # We'll plot the initial condition for network and empty lines for time series
    # so we clamp i_step so we don't go out of range
    i_show = max(0, min(i_step, steps - 1))

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
    ax_net = fig.add_subplot(gs_top[0])
    ax_ts = fig.add_subplot(gs_top[1])

    num_cols = 6
    num_rows = int(np.ceil(N / num_cols))
    gs_bottom = gridspec.GridSpecFromSubplotSpec(num_rows, num_cols, subplot_spec=gs[1])

    activation_axes = []

    # --- Network plot ---
    pos = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=V[i_show],
        cmap="coolwarm",
        node_size=500,
        vmin=-80, vmax=40,
        ax=ax_net
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax_net)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax_net)
    cbar = plt.colorbar(nodes, ax=ax_net, label="Membrane Voltage (mV)")
    ax_net.set_title(f"Network: Step={i_show} (t = {i_show*dt:.1f} ms)")

    # --- Time series plot ---
    ax_ts.set_title("Average Network Voltage Over Time")
    ax_ts.set_xlabel("Time (ms)")
    ax_ts.set_ylabel("Avg Voltage (mV)")
    ax_ts.set_xlim(0, time_array[-1])
    ax_ts.set_ylim(V.min() - 10, V.max() + 10)

    # Plot average voltage up to current step
    ax_ts.plot(time_array[:i_show], avg_V[:i_show], color='blue', lw=2)

    # --- Bottom subplots for each neuron's voltage vs. time ---
    for idx, spec in enumerate(gs_bottom):
        ax = fig.add_subplot(spec)
        if idx < N:
            activation_axes.append(ax)
            ax.set_title(f"Neuron {idx}", fontsize=8)
            ax.set_xlim(0, time_array[-1])
            ax.set_ylim(V.min(), V.max())
            ax.tick_params(axis='both', labelsize=6)

            # Plot neuron's voltage up to the current step
            ax.plot(time_array[:i_show], V[:i_show, idx], color='blue', lw=1)
            ax.axvline(x=i_show*dt, color='red', lw=1)
        else:
            ax.axis('off')

    plt.tight_layout()
    return fig


# ---------------------------
# The Streamlit Real-Time Page
# ---------------------------

def main():
    st.title("Real-Time Izhikevich + Gap-Junction Simulation in Streamlit")
    st.markdown("""
    This demo incrementally performs each time step of the simulation, updating the plots in 
    a loop to approximate a real-time animation inside Streamlit.
    \n
    **Warning**: This can be slow if `tmax` and `steps` are large. For a smoother experience:
    - Lower `tmax` or increase `dt`.
    - Reduce the size of the network `N`.
    """)

    # Sidebar for graph type
    st.sidebar.header("Network Parameters")
    graph_type = st.sidebar.selectbox(
        "Graph Type",
        ("erdos-renyi", "powerlaw", "ladder", "star", "watts-strogatz", "complete", "cycle", "path", "grid2d"),
        index=7  # default "path"
    )
    p = st.sidebar.slider("Probability p (Erdos-Renyi, Watts-Strogatz)", 0.0, 1.0, 0.2, 0.01)
    m = st.sidebar.slider("m (Barabási–Albert)", 1, 10, 3, 1)
    k = st.sidebar.slider("k (Watts-Strogatz)", 2, 12, 6, 1)

    # If using grid2d
    st.sidebar.markdown("**Grid2D Dimensions** (only used if graph_type='grid2d')")
    rows = st.sidebar.number_input("Rows", min_value=2, max_value=20, value=5)
    cols = st.sidebar.number_input("Cols", min_value=2, max_value=20, value=5)
    grid_dims = (rows, cols)

    # Simulation run parameters
    tmax = st.sidebar.slider("Total Simulation Time (ms)", 100, 2000, 500, 50)
    dt = st.sidebar.slider("Time Step (ms)", 0.1, 5.0, 0.4, 0.1)

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if st.button("Initialize / Reset"):
        initialize_simulation(
            N=24, dt=dt, tmax=tmax,
            graph_type=graph_type,
            p=p, m=m, k=k,
            grid_dims=grid_dims
        )
        st.session_state.initialized = True
        st.success("Simulation initialized. Now click 'Start Real-Time Simulation'.")

    if st.session_state.initialized:
        if st.button("Start Real-Time Simulation"):
            st.info("Running real-time steps. This may take a while...")

            placeholder = st.empty()

            # We'll keep stepping until we reach the last step
            while st.session_state.current_step < st.session_state.steps - 1:
                simulation_step()         # do one time step
                fig = plot_current_state()# plot current state
                placeholder.pyplot(fig)   # update the figure in the placeholder
                # Sleep to emulate "real-time" pacing
                time.sleep(0.02)

            st.success("Simulation completed!")
            # Show final figure
            final_fig = plot_current_state()
            placeholder.pyplot(final_fig)

    else:
        st.warning("Press 'Initialize / Reset' to set up the simulation first.")


if __name__ == "__main__":
    main()
