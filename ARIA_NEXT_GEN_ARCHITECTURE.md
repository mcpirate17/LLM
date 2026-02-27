# ARIA Next-Generation Architecture: The Path to Post-Transformer Supremacy

**Implementation Status:**
- [x] [C:claude-opus 2026-02-26] §1.1 CliffordLinear — `research/mathspaces/clifford.py`
- [x] [C:claude-opus 2026-02-26] §1.2 PoincareDistanceRouting — `research/mathspaces/hyperbolic.py`
- [x] [C:claude-opus 2026-02-26] §1.3 PersistentHomologyFilter — `research/mathspaces/tda.py`
- [x] [C:claude-opus 2026-02-26] §2 ScalingPredictor (NTK) — `research/eval/scaling_predictor.py`
- [x] [C:claude-opus 2026-02-26] §3 DifferentiableDAG — `research/search/differentiable_dag.py`
- [x] [C:claude-opus 2026-02-26] §4 KernelAgent — `research/synthesis/kernel_agent.py`

## 1. Mathematical Primitive Expansion (Beyond Linear Algebra)

To achieve a 3x-5x leap in parameter/FLOP efficiency over Attention and SSMs, ARIA must explore mathematical spaces that naturally encode complex relationships (hierarchies, multi-vector interactions, topological features) more efficiently than flat Euclidean spaces.

### 1.1 Clifford Algebras (Geometric Algebra)
**Theory:** Clifford algebras generalize complex numbers and quaternions to arbitrary dimensions, allowing operations on scalars, vectors, bivectors (oriented areas), and higher-grade multivectors. This enables a single operation to capture rotations, reflections, and projections simultaneously, providing a richer representation of interactions than standard matrix multiplication.

**PyTorch Implementation (`primitives.py`):**
```python
import torch
import torch.nn as nn

class CliffordLinear(nn.Module):
    """
    A simplified Clifford Algebra (Geometric Algebra) linear layer.
    Operates on multivectors (scalar, vector, bivector, pseudoscalar) in 3D space.
    Input shape: (batch, seq_len, dim, 8) where 8 is the dimension of the Cl(3,0) algebra.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Geometric product weights
        self.weight = nn.Parameter(torch.randn(dim, 8, 8) / (8 ** 0.5))
        self.bias = nn.Parameter(torch.zeros(dim, 8))

    def forward(self, x):
        # x: (B, L, D, 8)
        # Geometric product is bilinear, we approximate with a learned tensor contraction
        # out_i = sum_{j,k} W_{i,j,k} x_j x_k (simplified to linear transform over multivector components)
        # For true geometric product, the interaction rules are fixed, but we learn a projection
        # that respects the multivector structure.
        out = torch.einsum('bldi,dij->bldj', x, self.weight) + self.bias
        return out
```

### 1.2 Hyperbolic Geometry (Poincaré Ball)
**Theory:** Hyperbolic space has exponential volume growth, making it ideal for embedding hierarchical data (like language trees or knowledge graphs) with minimal distortion.

**PyTorch Implementation (`primitives.py`):**
```python
class PoincareDistanceRouting(nn.Module):
    """
    Routes information based on hyperbolic distance in the Poincaré ball model.
    """
    def __init__(self, dim, c=1.0):
        super().__init__()
        self.c = c # Curvature
        self.centroids = nn.Parameter(torch.randn(8, dim) * 1e-3) # 8 routing heads

    def mobius_add(self, x, y):
        # Simplified Möbius addition
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        den = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        return num / den.clamp_min(1e-15)

    def forward(self, x):
        # x: (B, L, D)
        # Compute hyperbolic distance to centroids
        B, L, D = x.shape
        x_expanded = x.unsqueeze(2) # (B, L, 1, D)
        c_expanded = self.centroids.unsqueeze(0).unsqueeze(0) # (1, 1, H, D)
        
        # Distance: arcosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))
        diff = self.mobius_add(-x_expanded, c_expanded)
        dist = 2 * torch.atanh(torch.sqrt(torch.tensor(self.c)) * torch.norm(diff, dim=-1).clamp_max(1 - 1e-5)) / torch.sqrt(torch.tensor(self.c))
        
        routing_weights = torch.softmax(-dist, dim=-1) # (B, L, H)
        return routing_weights
```

### 1.3 Topological Data Analysis (TDA) Layer
**Theory:** TDA extracts invariant shape features (connected components, holes, voids) from data manifolds. A differentiable TDA layer can filter out noise and focus on the persistent topological structure of the sequence.

**PyTorch Implementation (`primitives.py`):**
```python
class PersistentHomologyFilter(nn.Module):
    """
    Approximates a topological filter by computing local distance matrices
    and extracting spectral features (eigenvalues of the graph Laplacian).
    """
    def __init__(self, k_neighbors=5):
        super().__init__()
        self.k = k_neighbors

    def forward(self, x):
        # x: (B, L, D)
        # Compute pairwise distances in sequence
        dist = torch.cdist(x, x) # (B, L, L)
        
        # k-NN graph adjacency
        topk, indices = torch.topk(dist, self.k, largest=False, dim=-1)
        A = torch.zeros_like(dist).scatter_(-1, indices, 1.0)
        A = (A + A.transpose(-1, -2)) / 2 # Symmetrize
        
        # Graph Laplacian
        D = torch.diag_embed(A.sum(dim=-1))
        L_graph = D - A
        
        # Spectral features (approximate topological invariants)
        # We use power iteration or simply the diagonal of L for a fast differentiable proxy
        spectral_proxy = torch.diagonal(L_graph, dim1=-2, dim2=-1).unsqueeze(-1)
        
        return x * torch.sigmoid(spectral_proxy) # Modulate features based on local topology
```

## 2. The "Micro-to-Macro" Proxy Predictor

To overcome the "Bitter Lesson," ARIA needs to predict the scaling laws of an architecture from micro-training runs.

### 2.1 Neural Tangent Kernel (NTK) Condition Number
**Theory:** The NTK describes the evolution of neural networks during gradient descent in the infinite-width limit. A well-conditioned NTK implies stable, scale-invariant learning. If the NTK condition number degrades rapidly with depth or width, the architecture will not scale.

### 2.2 Maximal Update Parametrization ($\mu$P) Transferability
**Theory:** $\mu$P ensures that optimal hyperparameters transfer across model sizes. ARIA can test if an architecture obeys $\mu$P scaling rules. If the optimal learning rate shifts unpredictably between a 1M and 10M parameter model, it will fail at 10B.

### 2.3 Implementation (`eval/scaling_predictor.py`)
```python
import torch
import torch.nn as nn

class ScalingPredictor:
    def __init__(self, model_fn, dataloader):
        self.model_fn = model_fn
        self.dataloader = dataloader

    def compute_ntk_condition_number(self, model, x):
        """Computes the condition number of the empirical NTK."""
        model.eval()
        params = list(model.parameters())
        
        # Compute Jacobian J = dy/dtheta
        y = model(x)
        J = []
        for i in range(y.shape[0]):
            model.zero_grad()
            y[i].backward(retain_graph=True)
            grads = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
            J.append(grads)
        J = torch.stack(J)
        
        # NTK = J @ J^T
        ntk = J @ J.T
        eigenvalues = torch.linalg.eigvalsh(ntk)
        cond_num = eigenvalues[-1] / (eigenvalues[0] + 1e-8)
        return cond_num.item()

    def evaluate_scaling_ceiling(self):
        """
        Evaluates the architecture at two micro-scales to predict macro-scaling.
        """
        model_small = self.model_fn(dim=64, depth=2)
        model_med = self.model_fn(dim=128, depth=4)
        
        x, _ = next(iter(self.dataloader))
        
        cond_small = self.compute_ntk_condition_number(model_small, x)
        cond_med = self.compute_ntk_condition_number(model_med, x)
        
        # If condition number grows exponentially, scaling ceiling is low
        growth_rate = cond_med / (cond_small + 1e-8)
        
        # Score: 1.0 is perfect scaling, 0.0 is catastrophic failure
        scaling_score = max(0.0, 1.0 - 0.1 * growth_rate)
        return scaling_score
```

## 3. Continuous/Differentiable Graph Topology

Discrete evolutionary search is inefficient. We can use a continuous relaxation of the architecture space, inspired by DARTS (Differentiable Architecture Search), but adapted for heterogeneous computational graphs.

### 3.1 Differentiable Routing Matrix
Instead of hard-wiring Node A to Node B, every node outputs to a shared "blackboard" or routing matrix. The input to Node B is a weighted sum of all previous nodes' outputs, where the weights are learned architecture parameters ($\alpha$).

### 3.2 Gumbel-Softmax for Hard Routing
During search, we use the Gumbel-Softmax trick to sample discrete connections while maintaining differentiability. As temperature $\tau \to 0$, the routing becomes a hard DAG.

### 3.3 Implementation Concept
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableDAG(nn.Module):
    def __init__(self, num_nodes, dim, operations):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.ops = nn.ModuleList(operations) # List of heterogeneous ops
        
        # Architecture parameters: alpha[i, j] is the weight of edge from node j to node i
        # i > j to ensure DAG property
        self.alpha = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
    def forward(self, x, temperature=1.0):
        # Node 0 is the input
        node_outputs = [x]
        
        for i in range(1, self.num_nodes):
            # Compute routing weights from previous nodes to node i
            # Mask out future nodes to enforce DAG
            mask = torch.zeros(self.num_nodes).to(x.device)
            mask[:i] = 1.0
            
            logits = self.alpha[i] * mask
            logits = logits.masked_fill(mask == 0, float('-inf'))
            
            # Differentiable sampling of connections
            weights = F.gumbel_softmax(logits[:i], tau=temperature, hard=False)
            
            # Aggregate inputs
            node_input = sum(w * out for w, out in zip(weights, node_outputs))
            
            # Apply operation for node i
            node_output = self.ops[i](node_input)
            node_outputs.append(node_output)
            
        # Final output is the last node
        return node_outputs[-1]
```

## 4. Autonomous Custom Kernel Generation

To compete with Mamba's hardware efficiency, ARIA must compile its discovered mathematical graphs into fused GPU kernels.

### 4.1 The Agentic Workflow
1. **Graph Extraction:** ARIA identifies a high-performing subgraph (e.g., a Clifford projection followed by a hyperbolic routing).
2. **LLM Translation:** ARIA prompts an LLM (e.g., Claude/GPT-4) with the PyTorch code and asks for a fused Triton kernel.
3. **Compilation & Verification:** ARIA compiles the Triton kernel and runs a fuzzing test against the PyTorch reference to ensure numerical equivalence (within FP16/BF16 tolerances).
4. **Profiling:** ARIA runs the kernel through PyTorch Profiler to measure SRAM usage, memory bandwidth, and FLOPs.
5. **Iterative Refinement:** If the kernel is slower than the PyTorch baseline or fails verification, the error trace is fed back to the LLM for debugging.

### 4.2 Workflow Implementation Outline
```python
import torch
import triton
import subprocess

class KernelAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def generate_triton_kernel(self, pytorch_code, math_description):
        prompt = f"""
        Convert the following PyTorch module into a highly optimized, fused Triton kernel.
        Maximize SRAM usage and minimize global memory reads.
        
        PyTorch Code:
        {pytorch_code}
        
        Mathematical Context:
        {math_description}
        
        Output ONLY valid Python code containing the Triton @triton.jit kernel and a PyTorch wrapper function.
        """
        return self.llm.generate(prompt)

    def verify_and_profile(self, triton_code, pytorch_module, input_shape):
        # 1. Dynamically execute the generated code
        local_env = {}
        try:
            exec(triton_code, globals(), local_env)
            triton_wrapper = local_env['triton_wrapper']
        except Exception as e:
            return False, f"Compilation Error: {e}"

        # 2. Verification
        x = torch.randn(input_shape, device='cuda', dtype=torch.float16)
        try:
            y_ref = pytorch_module(x)
            y_triton = triton_wrapper(x)
            if not torch.allclose(y_ref, y_triton, atol=1e-3):
                return False, "Numerical mismatch."
        except Exception as e:
            return False, f"Runtime Error: {e}"

        # 3. Profiling
        # Use triton.testing.do_bench to measure execution time
        ms_pytorch = triton.testing.do_bench(lambda: pytorch_module(x))
        ms_triton = triton.testing.do_bench(lambda: triton_wrapper(x))
        
        speedup = ms_pytorch / ms_triton
        return True, {"speedup": speedup, "ms_triton": ms_triton}
```

## Conclusion
By integrating non-Euclidean mathematical primitives, NTK-based scaling predictors, differentiable DAG topologies, and LLM-driven Triton kernel fusion, ARIA will transition from a heuristic search tool into a rigorous, hardware-aware meta-architect capable of discovering the next generation of foundation models.
