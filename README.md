# principled-markovian-transformer

A hybrid sequential modeling architecture that bridges probabilistic Markovian dynamics with attention-based Transformers through a theoretically grounded Mixture-of-Experts (MoE) fusion.

🔑 Key Contributions

Multi-Expert Transition Module (METM): Specialized experts capture short-, medium-, long-range, and global temporal patterns.

Data-Dependent Routing: Adaptive MoE routing dynamically selects experts based on input characteristics, ensuring load balancing.

Principled Fusion Architecture: Conflict-aware integration of Markovian transitions and self-attention with formal convergence guarantees.

Theoretical Foundations: Formal proofs for convergence rates, universal approximation properties, and complexity bounds.

State-of-the-Art Performance:

RULER: 94.2% average accuracy (+5.1 over Jamba)

InfiniBench: 89.7% accuracy on 100K+ token tasks

35% memory efficiency improvement over standard Transformers at 64K sequences.

📂 Repository Structure
.
├── dataset_factory.py      # Dataset creation and preprocessing utilities
├── evaluation_framework.py # Benchmarking on RULER, InfiniBench, and custom tasks
├── markov_transformer.py   # Core architecture: METM, routing, fusion layers
├── run_experiments.py      # Scripts to launch training & evaluation experiments
├── train.py                # Training loop and optimization routines
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules


Training

Train the base Principled Markovian Transformer on a supported dataset:

python train.py --config configs/training/standard_training.yaml

Running Experiments

Reproduce paper results with a single command:

python run_experiments.py --benchmark RULER --context 32K

⚡ Benchmarks
Benchmark	Metric	Result
RULER	Retrieval/Multi-hop/Aggregation (32K)	94.2%
InfiniBench	Extreme long-context (100K+)	89.7%
Memory Efficiency	64K sequence	35% less vs. standard Transformer
