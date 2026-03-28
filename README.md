# 🐉 BDH Sparse Brain Visualizer

> **Post-Transformer Hackathon · IIT Ropar × Pathway**  
> Path A: Visualization & Inner Worlds

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Space-orange?style=for-the-badge)](https://huggingface.co/spaces/YOUR_USERNAME/bdh-sparse-brain)
[![Paper](https://img.shields.io/badge/📄_Paper-arXiv_2509.26507-blue?style=for-the-badge)](https://arxiv.org/abs/2509.26507)
[![Repo](https://img.shields.io/badge/⚙️_BDH_Repo-pathwaycom/bdh-green?style=for-the-badge)](https://github.com/pathwaycom/bdh)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 🧠 What We Built

An **interactive visualization** that makes the Dragon Hatchling (BDH) architecture viscerally understandable — revealing properties that transformers fundamentally cannot show.

The Transformer Explainer by Georgia Tech became the standard for understanding transformers. **BDH Sparse Brain** aims to do the same for the post-transformer era.

### The Core Insight

| | Transformer | BDH |
|---|---|---|
| Neurons fired per token | ~95–100% | **~5% (sparse!)** |
| Memory | KV-cache grows forever | **Constant size (Hebbian)** |
| Attention | O(T²) quadratic | **O(T) linear** |
| Interpretable | ✗ Black box | **✓ Monosemantic synapses** |
| Graph visualizable | ✗ Dense matrix | **✓ Literal force graph** |

---

## 🎬 Demo Preview

> **[👉 Click here for the Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/bdh-sparse-brain)**

The visualizer has **4 interactive sections**:

### Tab 1 — Sparsity Comparison
Type any token → BDH fires ~5% neurons vs Transformer's ~95%.  
"London" specifically activates *currency synapses* and *country synapses* — monosemantic encoding you can point to and explain.

### Tab 2 — Hebbian Learning Animator
Watch the σ (synaptic state) matrix evolve as tokens are processed.  
Synapses visibly strengthen when neurons co-activate. This is **inference-time learning** — no backpropagation, no retraining.

### Tab 3 — Scale-Free Graph Topology  
BDH's G_x = E @ D_x rendered as an interactive force graph.  
Hub neurons emerge spontaneously from training — like how real brains organize. Click any node to trace its connections.

### Tab 4 — Architecture Comparison  
Side-by-side breakdown with real benchmark numbers.

---

## 📁 Repository Structure

```
bdh-sparse-brain/
│
├── index.html              ← Main visualizer (single-file, deployable anywhere)
│
├── src/
│   ├── bdh_instrument.py   ← How to add activation logging to bdh.py
│   └── export_activations.py ← Export real model activations to JSON
│
├── notebooks/
│   └── bdh_sparsity_demo.ipynb ← Google Colab notebook (run BDH, export data)
│
├── docs/
│   ├── ARCHITECTURE.md     ← Deep dive into BDH internals
│   └── METHODOLOGY.md      ← How we measured and visualized sparsity
│
├── assets/
│   └── (screenshots, demo GIFs go here)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Just open the demo
```bash
git clone https://github.com/YOUR_USERNAME/bdh-sparse-brain
cd bdh-sparse-brain
# Open index.html in any browser — no server needed
open index.html
```

### Option 2 — Run with real BDH model activations

**Step 1: Install BDH**
```bash
git clone https://github.com/pathwaycom/bdh
cd bdh
pip install torch numpy
```

**Step 2: Instrument the model**
```bash
# Copy our instrumentation patch
cp ../bdh-sparse-brain/src/bdh_instrument.py .
python bdh_instrument.py
```

**Step 3: Export activations**
```bash
python src/export_activations.py --checkpoint out/ckpt.pt --output assets/activations.json
```

**Step 4: Load in visualizer**

Open `index.html` and in the browser console:
```js
// The visualizer auto-loads assets/activations.json if present
loadRealActivations('assets/activations.json')
```

---

## 🧪 What This Reveals About BDH

### 1. Sparsity is Architectural, Not Regularized
BDH's ~5% activation rate comes from its **ReLU-lowrank design** — not from L1 penalties, pruning, or distillation. The sparse firing emerges naturally from `G_x = ReLU(E @ D_x)` where most entries are zero.

### 2. Monosemantic Synapses
The paper (Section 6.3) demonstrates "currency synapses" that activate consistently for GBP, USD, EUR across languages, and "country synapses" for nation names. Our visualizer shows this with token-level granularity.

### 3. Constant Memory at Any Context Length
BDH's σ matrix is `O(n×d)` — fixed size regardless of sequence length. Community experiments (see `glass-brain`) have demonstrated **50,000+ tokens with flat memory** while transformers OOM at ~12k on a T4 GPU.

### 4. Hebbian Learning at Inference Time
`σ_{t+1} = σ_t + η · (pre ⊗ post)` — synapses strengthen when neurons co-activate. No gradient computation. No backpropagation. The model you deploy **gets smarter as it runs**.

---

## 📊 Benchmark Numbers Referenced

| Task | Transformer | BDH | Notes |
|------|-------------|-----|-------|
| Language modeling | GPT-2 baseline | Competitive | Section 4.2 of paper |
| Sentiment (finance) | DistilBERT 67M | **BDH 14M** | 14M params beats 67M |
| CIFAR-10 | ViT-Tiny 74% (5.7M) | **79.54% (3.2M)** | Fewer params, higher accuracy |
| Memory at 50k tokens | ❌ OOM crash | **✓ Flat usage** | T4 GPU, glass-brain project |

---

## 🛠️ Technical Decisions

**Why a single HTML file?**  
Zero setup for judges and community. Open in browser → works instantly. Deployable to HuggingFace Spaces as a static site with one file upload.

**Why simulate activations (not real model)?**  
The visualizer is architected to accept real activation JSON (see `src/export_activations.py`). The simulated data closely mirrors real BDH sparsity distributions measured in the paper (~5% BDH, ~95% transformer). For a production version, replace with checkpoint-generated data.

**Why Canvas + vanilla JS over D3?**  
Maximum compatibility — runs on any device, no CDN dependency, instant load time. The Hebbian animator and graph topology require frame-level control best served by Canvas2D.

---

## 🔬 Methodology

Full details in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). In brief:

1. **Sparsity measurement**: Count `(x > 0)` after each ReLU activation layer in `bdh.py`. Average across sequence positions and heads.
2. **Synapse strength**: Track σ matrix norms per synapse pair across token sequence.
3. **Graph topology**: Extract adjacency from `E @ D_x` at final layer, threshold by activation magnitude, render as force-directed graph.
4. **Transformer baseline**: Count non-negligible softmax attention outputs (> 0.01 threshold) per head as proxy for "active" neurons.

---

## ⚠️ Limitations & Future Scope

- **Simulated vs real activations**: Current version uses architecture-accurate simulations. Next step: hook real checkpoint inference.
- **Static graph**: The topology visualizer uses a scale-free graph approximation. Real G_x topology extraction requires layer-by-layer forward pass instrumentation.
- **Single layer shown**: Sparsity comparison shows one representative layer. A multi-layer scrubber is planned.
- **No quantitative monosemanticity score**: We show qualitative monosemanticity. A systematic synapse-concept correlation score (like TMI/SPS metrics from Pragadhishnitt's work) would strengthen this.

---

## 🙌 TripleIT Titans

| Name | Role | Institute |
|------|------|-----------|
| Vanshiv Garg | Visualization, Frontend | IIIT UNA |
| Abhishek Kumar Sinha | BDH Instrumentation, ML | IIIT UNA |
| Neev Bolia | Architecture Research | IIIT UNA |
| Sujeet | Frontend Helper | IIIT UNA |

---

## 📚 References & Resources

- **BDH Paper**: [The Dragon Hatchling — arXiv 2509.26507](https://arxiv.org/abs/2509.26507)
- **Official Repo**: [pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- **Pathway**: [pathway.com](https://pathway.com)
- **Visualization Inspiration**: [Transformer Explainer (Georgia Tech)](https://poloclub.github.io/transformer-explainer)
- **Community Reference**: [krychu/bdh](https://github.com/krychu/bdh) — pathfinding visualization
- **Memory Scaling Demo**: [glass-brain](https://github.com/sharmilcd/glass-brain)

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
  <sub>Built for the Post-Transformer Hackathon by Pathway · IIT Ropar · 2025</sub><br>
  <sub>⭐ Star <a href="https://github.com/pathwaycom/bdh">pathwaycom/bdh</a> · 👍 Upvote on <a href="https://huggingface.co/papers/2509.26507">HuggingFace</a></sub>
</div>
