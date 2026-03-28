# Visualization Methodology

> How the BDH Sparse Brain Visualizer measures, simulates, and presents BDH's architectural properties.

---

## Tab 1: Sparsity Comparison

### What We're Measuring

**BDH sparsity**: The fraction of neurons with activation > 0 after each ReLU layer.  
**Transformer density**: The fraction of attention weights above a threshold (> 0.01) per head.

### Measurement Approach

```python
# In bdh.py forward pass, after each ReLU:
sparsity = (activation > 0).float().mean().item()
# Typical result: 0.03 - 0.07 (3-7%)
```

```python
# In GPT forward pass, after softmax attention:
density = (attention_weights > 0.01).float().mean().item()
# Typical result: 0.88 - 0.96 (88-96%)
```

### Current Implementation

The visualizer uses **architecture-accurate simulation** based on the paper's reported values (§6.4):
- BDH: uniform distribution in [0.03, 0.07] per token, anchored to concept type
- Transformer: uniform distribution in [0.88, 0.96] 

**Upgrade path**: Replace with real checkpoint data using `src/export_activations.py`.

### Monosemanticity Representation

Token categories map to specific "synapse groups" in the simulation:

| Token Type | BDH Behavior | Visual |
|-----------|--------------|--------|
| Currency (dollar, euro) | Currency synapse cluster fires | Same neuron subset lights up consistently |
| Country (France, London) | Country synapse cluster fires | Overlapping but distinct subset |
| Function words (the, is) | Very few synapses fire | <3% activation |
| Random text | Domain-specific spread | ~5% random spread |

This matches the paper's Section 6.3 findings on monosemantic synapses.

---

## Tab 2: Hebbian Learning Animator

### What We're Simulating

The σ (sigma) matrix update rule from Equation (8) of the paper:

```
σ_{t+1} = σ_t + η · (pre ⊗ post)
```

Where `pre = ReLU(E·x_t)` and `post = ReLU(D·x_t)`.

### Visual Representation

- **Neurons** = circles, radius proportional to cumulative synaptic strength
- **Synapses** = lines, thickness proportional to σ weight
- **Firing** = amber highlight on currently active neurons and their connections
- **Strengthening** = gradual increase in line thickness and node radius

### What the Animation Shows

1. **Token arrives** → ~5% of neurons activate (amber flash)
2. **Co-activation** → pairs that fire together get their synapse weight increased
3. **Over time** → frequently co-activated pairs develop thick bright connections
4. **Rarely co-activated** → connections slowly decay

This is Hebbian plasticity: "neurons that fire together, wire together."

### Why This Is Impossible for Transformers

Transformers have no equivalent of σ. Their weights are:
- Fixed at inference time
- Updated only via backpropagation (offline, requires labeled data)
- Stored in weight matrices, not as a dynamic state

BDH's σ is a **living state** that evolves with every token.

---

## Tab 3: Scale-Free Graph Topology

### What We're Showing

BDH's G_x = E @ D_x is a real adjacency matrix. We visualize the **degree distribution** of the resulting graph, which follows a power law (hence "scale-free").

### Graph Generation Method

We use **preferential attachment** (Barabási–Albert model) to approximate the scale-free topology that emerges in BDH:

```python
# Each new node attaches to existing nodes with probability ∝ degree
P(connect to node i) = degree(i) / sum(all degrees)
```

This produces the hub-and-spoke structure observed in BDH's empirical analysis (§5.4).

### Power Law Verification

Real BDH graphs follow: `P(k) ∝ k^{-γ}` where γ ≈ 2-3 (typical for scale-free networks).

This same distribution appears in:
- Biological neural networks (the brain)
- The World Wide Web
- Protein interaction networks
- Social networks

**Paper reference**: Section 5.4 — "Emergence of modularity and scale-free structure"

### Hub Neurons

Nodes with degree > 50th percentile are classified as "hubs" and rendered larger.  
These correspond to neurons that participate in many concepts — the "generalist" neurons in BDH's architecture.

---

## Tab 4: Architecture Comparison

### Data Sources

All benchmark numbers in the comparison table come directly from the BDH paper:

| Claim | Source |
|-------|--------|
| ~5% activation rate | §6.4, Figure 8 |
| Competitive with GPT-2 | §4.2, Table 1 |
| 50k+ tokens flat memory | Community: glass-brain project |
| 14M BDH > 67M DistilBERT | Pragadhishnitt hackathon project |
| 79.54% CIFAR-10 | takzen/vision-bdh |
| Monosemantic synapses | §6.3 |
| Model merging by concatenation | §7.1 |

---

## Limitations

### What We Know Is Approximate

1. **Token → synapse mapping**: The specific neurons shown for "London" are simulated to match the paper's concept of currency/country synapses. The actual neuron indices depend on the specific trained checkpoint.

2. **Graph topology**: The force-directed graph uses an approximation of scale-free structure. Real G_x extraction requires layer-by-layer instrumented forward passes.

3. **Transformer baseline**: ~94% density is the reported mean. Individual tokens and attention heads vary.

### Upgrade Path to Real Data

```bash
# 1. Train BDH on your hardware
cd pathwaycom/bdh && python train.py

# 2. Export real activations
python src/export_activations.py --checkpoint out/ckpt.pt

# 3. Place in assets/
cp activations_output.json bdh-sparse-brain/assets/activations.json

# 4. Visualizer auto-loads real data
# Open index.html → Tab 1 now shows real neuron firing
```

---

## Future Work

- [ ] Real checkpoint instrumentation (upgrade from simulation)
- [ ] Multi-layer scrubber (show all layers, not just representative one)
- [ ] Monosemanticity score (quantitative: synapse-concept correlation)
- [ ] Memory usage live demo (show σ size vs KV-cache size over tokens)
- [ ] Side-by-side inference video (same sentence, BDH vs GPT, side by side)
- [ ] 3D graph topology using Three.js
