import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";

// NetworkGraph component for visualizing the network structure
const NetworkGraph = ({ nodes, edges }) => {
  const [nodePositions, setNodePositions] = useState({});
  const [dimensions] = useState({ width: 600, height: 400 });

  useEffect(() => {
    if (!nodes.length) return;
    let positions = {};
    nodes.forEach(node => {
      positions[node.id] = {
        x: Math.random() * dimensions.width,
        y: Math.random() * dimensions.height
      };
    });
    setNodePositions(positions);
  }, [nodes]);

  if (!nodes.length) return null;

  return (
    <svg 
      className="w-full h-full"
      viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
    >
      {edges.map((edge, i) => (
        <line
          key={`edge-${i}`}
          x1={nodePositions[edge.source]?.x}
          y1={nodePositions[edge.source]?.y}
          x2={nodePositions[edge.target]?.x}
          y2={nodePositions[edge.target]?.y}
          stroke="#9CA3AF"
          strokeWidth="1"
          opacity="0.6"
        />
      ))}
      {nodes.map(node => (
        <g
          key={`node-${node.id}`}
          transform={`translate(${nodePositions[node.id]?.x}, ${nodePositions[node.id]?.y})`}
        >
          <circle r="6" fill="#3B82F6" stroke="white" strokeWidth="2" />
          <text dx="8" dy="4" className="text-sm fill-current">{node.label}</text>
        </g>
      ))}
    </svg>
  );
};

// Main simulation component
const NeuralNetworkSim = () => {
  // Network parameters
  const [networkType, setNetworkType] = useState("erdos-renyi");
  const [numNeurons, setNumNeurons] = useState(20);
  const [probability, setProbability] = useState(0.2);
  const [numNeighbors, setNumNeighbors] = useState(6);
  const [barabasiM, setBarabasiM] = useState(3);
  
  // Simulation parameters
  const [dt, setDt] = useState(0.4);
  const [tmax, setTmax] = useState(1000);
  
  // Results
  const [simulationData, setSimulationData] = useState([]);
  const [networkData, setNetworkData] = useState({ nodes: [], edges: [] });

  // Izhikevich model parameters
  const modelParams = {
    a: 0.02,
    b: 0.2,
    c: -50,
    d: 2
  };

  const createNetwork = () => {
    const nodes = Array.from({ length: numNeurons }, (_, i) => ({
      id: i,
      label: `N${i}`
    }));
    
    const edges = [];
    
    switch (networkType) {
      case "erdos-renyi":
        for (let i = 0; i < numNeurons; i++) {
          for (let j = i + 1; j < numNeurons; j++) {
            if (Math.random() < probability) {
              edges.push({ source: i, target: j });
            }
          }
        }
        break;
        
      case "powerlaw":
        for (let i = 0; i < barabasiM; i++) {
          for (let j = i + 1; j < barabasiM; j++) {
            edges.push({ source: i, target: j });
          }
        }
        for (let i = barabasiM; i < numNeurons; i++) {
          const degrees = new Array(i).fill(0);
          edges.forEach(edge => {
            if (edge.source < i) degrees[edge.source]++;
            if (edge.target < i) degrees[edge.target]++;
          });
          const totalDegree = Math.max(1, degrees.reduce((a, b) => a + b, 0));
          for (let m = 0; m < barabasiM; m++) {
            let r = Math.random() * totalDegree;
            let sum = 0;
            for (let j = 0; j < i; j++) {
              sum += degrees[j];
              if (sum > r) {
                edges.push({ source: i, target: j });
                break;
              }
            }
          }
        }
        break;
        
      case "watts-strogatz":
        for (let i = 0; i < numNeurons; i++) {
          for (let j = 1; j <= numNeighbors / 2; j++) {
            edges.push({ source: i, target: (i + j) % numNeurons });
          }
        }
        edges.forEach((edge, idx) => {
          if (Math.random() < probability) {
            let newTarget;
            do {
              newTarget = Math.floor(Math.random() * numNeurons);
            } while (newTarget === edge.source || edges.some(e => 
              (e.source === edge.source && e.target === newTarget) ||
              (e.source === newTarget && e.target === edge.source)
            ));
            edges[idx] = { source: edge.source, target: newTarget };
          }
        });
        break;
        
      case "complete":
        for (let i = 0; i < numNeurons; i++) {
          for (let j = i + 1; j < numNeurons; j++) {
            edges.push({ source: i, target: j });
          }
        }
        break;
        
      case "grid2d":
        const side = Math.ceil(Math.sqrt(numNeurons));
        for (let i = 0; i < numNeurons; i++) {
          const row = Math.floor(i / side);
          const col = i % side;
          if (col < side - 1 && i + 1 < numNeurons) edges.push({ source: i, target: i + 1 });
          if (row < side - 1 && i + side < numNeurons) edges.push({ source: i, target: i + side });
        }
        break;
    }
    
    return { nodes, edges };
  };

  const calculateGapCurrent = (V, i, network) => {
    const connectedNodes = network.edges
      .filter(e => e.source === i || e.target === i)
      .map(e => e.source === i ? e.target : e.source);
    return connectedNodes.reduce((sum, j) => {
      const g_gap = Math.random() * 0.2 + 0.1;
      return sum + g_gap * (V[j] - V[i]);
    }, 0);
  };

  const runSimulation = () => {
    const steps = Math.floor(tmax / dt);
    const N = numNeurons;
    const V = new Array(steps).fill(null).map(() => new Array(N).fill(-65.0));
    const U = new Array(N).fill(null).map((_, i) => modelParams.b * V[0][i]);
    const network = createNetwork();
    setNetworkData(network);
    
    const I_ext = new Array(N).fill(0);
    const stimulated = new Array(3).fill(null).map(() => Math.floor(Math.random() * N));
    stimulated.forEach(i => I_ext[i] = 20.0);
    
    for (let t = 0; t < steps - 1; t++) {
      for (let i = 0; i < N; i++) {
        const I_gap = calculateGapCurrent(V[t], i, network);
        const I_net = I_ext[i] + I_gap;
        const dVdt = 0.04 * V[t][i]**2 + 5 * V[t][i] + 140 - U[i] + I_net;
        const dUdt = modelParams.a * (modelParams.b * V[t][i] - U[i]);
        V[t + 1][i] = V[t][i] + dVdt * dt;
        U[i] += dUdt * dt;
        if (V[t + 1][i] >= 30) {
          V[t + 1][i] = modelParams.c;
          U[i] += modelParams.d;
        }
      }
    }
    
    setSimulationData(V.map((row, idx) => ({
      time: idx * dt,
      voltage: row.reduce((a, b) => a + b, 0) / N
    })));
  };

  useEffect(() => {
    const network = createNetwork();
    setNetworkData(network);
  }, [networkType, numNeurons, probability, numNeighbors, barabasiM]);

  return (
    <div className="p-4 max-w-7xl mx-auto space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Neural Network Simulation</CardTitle>
          <CardDescription>Izhikevich neuron model with gap junction coupling</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Network Type</label>
                <Select value={networkType} onValueChange={setNetworkType}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="erdos-renyi">Erdős-Rényi</SelectItem>
                    <SelectItem value="powerlaw">Power Law (Barabási-Albert)</SelectItem>
                    <SelectItem value="watts-strogatz">Watts-Strogatz</SelectItem>
                    <SelectItem value="complete">Complete</SelectItem>
                    <SelectItem value="grid2d">2D Grid</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <label className="text-sm font-medium">Number of Neurons</label>
                <Slider
                  value={[numNeurons]}
                  onValueChange={([value]) => setNumNeurons(value)}
                  min={5}
                  max={50}
                  step={1}
                  className="mt-2"
                />
                <span className="text-sm text-gray-600">{numNeurons}</span>
              </div>

              <div>
                <label className="text-sm font-medium">Time Step (dt)</label>
                <Slider
                  value={[dt]}
                  onValueChange={([value]) => setDt(value)}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  className="mt-2"
                />
                <span className="text-sm text-gray-600">{dt} ms</span>
              </div>

              <div>
                <label className="text-sm font-medium">Simulation Duration</label>
                <Slider
                  value={[tmax]}
                  onValueChange={([value]) => setTmax(value)}
                  min={100}
                  max={2000}
                  step={100}
                  className="mt-2"
                />
                <span className="text-sm text-gray-600">{tmax} ms</span>
              </div>
              
              {networkType === "erdos-renyi" && (
                <div>
                  <label className="text-sm font-medium">Connection Probability</label>
                  <Slider
                    value={[probability]}
                    onValueChange={([value]) => setProbability(value)}
                    min={0}
                    max={1}
                    step={0.05}
                    className="mt-2"
                  />
                  <span className="text-sm text-gray-600">{probability}</span>
                </div>
              )}

              {networkType === "watts-strogatz" && (
                <>
                  <div>
                    <label className="text-sm font-medium">Number of Neighbors (k)</label>
                    <Slider
                      value={[numNeighbors]}
                      onValueChange={([value]) => setNumNeighbors(value)}
                      min={2}
                      max={Math.min(10, numNeurons - 1)}
                      step={2}
                      className="mt-2"
                    />
                    <span className="text-sm text-gray-600">{numNeighbors}</span>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Rewiring Probability</label>
                    <Slider
                      value={[probability]}
                      onValueChange={([value]) => setProbability(value)}
                      min={0}
                      max={1}
                      step={0.05}
                      className="mt-2"
                    />
                    <span className="text-sm text-gray-600">{probability}</span>
                  </div>
                </>
              )}

              {networkType === "powerlaw" && (
                <div>
                  <label className="text-sm font-medium">New Edges per Node (m)</label>
                  <Slider
                    value={[barabasiM]}
                    onValueChange={([value]) => setBarabasiM(value)}
                    min={1}
                    max={5}
                    step={1}
                    className="mt-2"
                  />
                  <span className="text-sm text-gray-600">{barabasiM}</span>
                </div>
              )}
              
              <button
                onClick={runSimulation}
                className="w-full bg-blue-600 text-white rounded-lg px-4 py-2 hover:bg-blue-700"
              >
                Run Simulation
              </button>
            </div>
            
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={simulationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" label={{ value: 'Time (ms)', position: 'bottom' }} />
                  <YAxis label={{ value: 'Average Voltage (mV)', angle: -90, position: 'left' }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="voltage" stroke="#2563eb" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Network Structure</CardTitle>
        </CardHeader>
        <CardContent className="h-96 relative">
          <NetworkGraph nodes={networkData.nodes} edges={networkData.edges} />
        </CardContent>
      </Card>
    </div>
  );
};

export default NeuralNetworkSim;