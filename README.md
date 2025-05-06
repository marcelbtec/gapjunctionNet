# Neural Network Simulation with Gap Junctions

This project implements a real-time visualization of a neural network simulation using the Izhikevich neuron model with gap junction coupling. The simulation provides an interactive and comprehensive view of neural dynamics across different network topologies.

## Features

- **Multiple Network Topologies**: Supports various network configurations:
  - Erdos-Renyi (random)
  - Barabasi-Albert (scale-free)
  - Ladder
  - Star
  - Watts-Strogatz (small-world)
  - Complete
  - Cycle
  - Path
  - 2D Grid

- **Interactive Visualization**: Real-time display of:
  - Network topology with node voltage coloring
  - Average network voltage over time
  - Individual neuron activation traces
  - State space plot (V vs U)
  - Spike raster plot

- **Izhikevich Neuron Model**: Implements the Izhikevich neuron model with:
  - Gap junction coupling between neurons
  - External stimulation capabilities
  - Configurable neuron parameters

## Requirements

- Python 3.x
- NumPy
- NetworkX
- Matplotlib
- FFmpeg (for saving animations)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install numpy networkx matplotlib
```

## Usage

### Basic Usage

Run the simulation with default parameters:
```bash
python scripts/simulation_realtime.py
```

### Command Line Options

The script supports various command-line parameters to customize the simulation:

```bash
# Network parameters
--N N                    # Number of neurons (default: 20)
--graph-type TYPE        # Network topology (default: erdos-renyi)
--p P                    # Connection probability (default: 0.3)
--m M                    # Barabasi-Albert edges (default: 1)
--k K                    # Watts-Strogatz neighbors (default: 6)
--grid-dims ROWS COLS    # Grid dimensions (default: 5 4)

# Simulation parameters
--dt DT                  # Time step (default: 0.5)
--tmax TMAX             # Maximum simulation time (default: 600)

# Izhikevich model parameters
--a A                    # Parameter a (default: 0.02)
--b B                    # Parameter b (default: 0.2)
--c C                    # Parameter c (default: -50)
--d D                    # Parameter d (default: 2)

# Output options
--save                  # Save animation as video file
--output FILENAME       # Output filename (default: simulation.mp4)
--fps FPS              # Frames per second (default: 20)
--dpi DPI              # DPI for video (default: 300)
```

### Examples

Run a simulation with a Watts-Strogatz network:
```bash
python scripts/simulation_realtime.py --graph-type watts-strogatz --N 30 --k 4 --p 0.2
```

Run a simulation with a 2D grid and save the animation:
```bash
python scripts/simulation_realtime.py --graph-type grid2d --grid-dims 6 6 --save --output grid_simulation.mp4
```

Run a simulation with custom Izhikevich parameters:
```bash
python scripts/simulation_realtime.py --a 0.03 --b 0.25 --c -55 --d 2.5
```

## Visualization

The simulation provides a comprehensive visualization with four main panels:

1. **Network View**: Shows the network topology with nodes colored by their membrane potential
2. **Time Series**: Displays the average network voltage over time
3. **Neuron Activations**: Individual traces for each neuron
4. **State Space & Raster**: Shows the network's state space trajectory and spike timing

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 