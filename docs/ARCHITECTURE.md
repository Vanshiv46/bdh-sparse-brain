# BDH Architecture Deep Dive

> Reference document for the BDH Sparse Brain Visualizer project.  
> Post-Transformer Hackathon · IIT Ropar × Pathway

---

## 1. What Makes BDH Different

The Dragon Hatchling (BDH) departs from the transformer paradigm at the architectural level — not as an optimization, but as a replacement of the fundamental computation model.

### Transformer: Dense Matrix Operations
```
Every token → Every weight → Near-100% activation
Memory: KV-cache grows with sequence length (O(T))
Attention: O(T²) — quadratic scaling wall
```

### BDH: Sparse Graph Dynamics
```
Every token → Scale-free graph → ~5% activation
Memory: Hebbian synaptic state (O(n×d)) — constant forever
Attention: O(T) — linear, scales to infinite context
```

---

## 2. The Core Equation (BDH-GPU, §3.2)

The heart of BDH is the state-space formulation:

```
State update:
  σ_{t+1} = σ_t + η · ReLU(E·x_t) ⊗ ReLU(D·x_t)

Token output:
  y_t = W_out · ReLU(G_x · h_t)

Where:
  σ  = synaptic weight matrix (constant size regardless of T)
  η  = learning rate (Hebbian plasticity)
  E  = encoder matrix
  D  = decoder matrix
  G_x = E @ D_x  (the interpretable graph)
  h_t = hidden state at time t
```

This is Hebbian learning: `Δσ = pre-synaptic · post-synaptic activation`.  
The rule "neurons that fire together, wire together" is baked into the update equation.

---

## 3. Sparsity: Why ~5%?

BDH uses **ReLU activations** instead of SoftMax:

| Component | Transformer | BDH |
|-----------|-------------|-----|
| Attention | `SoftMax(QK^T/√d)` — all positive, sums to 1 | `ReLU(E @ D_x)` — most are exactly 0 |
| MLP | GELU — smooth, mostly non-zero | ReLU-lowrank — hard zeroes |
| Result | ~95% neurons receive signal | ~5% neurons fire |

SoftMax **cannot** produce exact zeroes — the exponential function is always positive.  
ReLU **does** produce exact zeroes — any negative input becomes exactly 0.

This is not a trick or regularization. It is a mathematical consequence of the activation function choice.

**Paper reference**: Section 6.4 — "Sparse positive activations"

---

## 4. The G_x Graph: Why It's Visualizable

The key matrix in BDH is:

```python
G_x = E @ D_x   # shape: [n_neurons, n_neurons]
```

This is a literal adjacency matrix of a neural graph. Each entry `G_x[i,j]` represents the connection strength from neuron `j` to neuron `i` for the current input `x`.

**Properties that emerge from this formulation:**
- **Scale-free topology**: Hub neurons appear spontaneously (few neurons with very high degree)
- **Modularity**: Neuron communities form around semantic concepts
- **Visualizability**: You can literally render it as a force-directed graph

Transformers have no equivalent — their computation is a dense matrix multiply with no interpretable graph structure.

**Paper reference**: Sections 2.2, 5.2, 5.4

---

## 5. Monosemantic Synapses (§6.3)

In transformers, individual neurons are **polysemantic** — they respond to multiple unrelated concepts. This is a known limitation (studied by Anthropic's interpretability team).

BDH synapses are **monosemantic** by design:

```
Currency synapse: Activates for: "dollar", "euro", "pound", "yen", "CHF"
                  Does NOT activate for: "London", "cat", "neural"

Country synapse:  Activates for: "France", "England", "Germany", "Japan"
                  Does NOT activate for: "dollar", "the", "network"
```

This behavior:
- Is consistent across languages (verified in the paper)
- Requires no special training objective
- Emerges from the Hebbian update rule

**Why?** Hebbian learning strengthens connections when neurons co-activate. Currency words consistently co-activate the same neurons → those synapses specialize for currency.

---

## 6. Constant Memory: The σ Matrix

The synaptic state σ has shape `[n_embd, n_embd]` — **fixed at model creation, never grows**.

Compare to transformer KV-cache:

```
Transformer KV-cache at T tokens:
  shape: [n_layers, 2, T, n_heads, head_dim]
  memory: grows linearly with T
  at T=50,000: ~6-8 GB on typical configs → OOM

BDH σ matrix at T tokens:
  shape: [n_embd, n_embd]
  memory: constant, independent of T
  at T=50,000: same as at T=1
```

**Experimentally demonstrated**: `glass-brain` project ran BDH to 50,000+ tokens on a T4 GPU (16GB VRAM) while the equivalent transformer crashed at ~12,000 tokens.

---

## 7. Inference-Time Learning

Standard neural networks require **backpropagation** to update weights — this means retraining on new data, which is expensive and offline.

BDH updates σ during the forward pass:

```python
# Pseudocode of the BDH-GPU forward pass
def forward(self, x_t, sigma):
    pre  = ReLU(self.E(x_t))   # pre-synaptic activation
    post = ReLU(self.D(x_t))   # post-synaptic activation
    
    # Hebbian update — happens during inference, no backprop needed
    sigma = sigma + self.eta * torch.outer(pre, post)
    
    # Use updated sigma for output
    g_x  = self.E @ (sigma @ self.D.T)
    y    = ReLU(g_x @ self.hidden)
    return y, sigma
```

The model you deploy **keeps learning** as it processes new tokens. No retraining. No fine-tuning. This is architecturally impossible for transformers.

---

## 8. Model Merging by Concatenation (§7.1)

Because BDH is a graph with fixed node types:

```python
# Train two specialists
model_fr = train(french_translation_data)
model_es = train(spanish_translation_data)

# Concatenate (not average — literal concatenation of graph nodes)
model_bilingual = concatenate(model_fr, model_es)
# → Works as a bilingual translator, no fine-tuning needed
```

This works because BDH's scale-free graph structure means the hub neurons in each specialist model don't interfere with each other when concatenated.

Transformer merging requires careful fine-tuning and often fails for distant tasks.

---

## 9. Implementation in bdh.py

Key locations in `pathwaycom/bdh/bdh.py`:

| Concept | Location in code |
|---------|-----------------|
| ReLU sparse activation | Look for `F.relu()` calls in the `BDHLayer.forward()` method |
| Hebbian state update | The `sigma` parameter update: `sigma = sigma + eta * (pre ⊗ post)` |
| G_x graph computation | `g_x = self.E @ D_x` — the interpretable adjacency matrix |
| Constant memory | `sigma` is passed as state, not KV-cache |

---

## 10. Further Reading

- **Full paper**: https://arxiv.org/abs/2509.26507
- **Paper HTML**: https://arxiv.org/html/2509.26507v1 (easier navigation)
- **Implementation**: https://github.com/pathwaycom/bdh/blob/main/bdh.py
- **Section guide**: See `docs/METHODOLOGY.md` for reading order
- **Community visualization**: https://github.com/krychu/bdh
- **Memory scaling demo**: https://github.com/sharmilcd/glass-brain
