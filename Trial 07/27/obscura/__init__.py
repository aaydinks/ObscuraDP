"""ObscuraDP Trial 07/27 -- the unified DP + FL + HE stack.

One federated pipeline in which differential privacy, federated learning and
homomorphic encryption all run at the same time, built by composing the three
prior ObscuraDP reproductions:

    Experiment 1 (arXiv:2508.10017)      non-IID Dirichlet clients, per-client
                                         SMOTETomek, FedProx, Opacus DP-SGD
    Experiment 2 (s41598-025-34536-9)    the controlled arm matrix, CKKS
                                         aggregation, payload/latency accounting
    Experiment 3 (arXiv:2303.10837)      chunked CKKS, Selective Parameter
                                         Encryption, O(n) overhead, losslessness

Module map:
    config        every hyperparameter (single source of truth)
    data          stroke + loan loading, scaling, non-IID/IID partitioning, SMOTE
    models        the FFNN/MLP family and flat<->state_dict conversion
    dp            sigma solving, RDP accounting, Opacus wiring, noise floors
    he            CKKS context, chunked encryption, naive + SPE aggregation
    sensitivity   clean and DP-aware gradient sensitivity, top-p masks
    fedprox       one client's local round (plaintext or DP-SGD, +/- proximal)
    metrics       held-out evaluation and multi-run summarisation
    aggregate     the single place plaintext / HE / SPE aggregation meet
    orchestrator  the full-stack federated loop that ties it all together
    bench         crypto and DP overhead harness (no training)
"""

__version__ = "1.0.0"
__trial__ = "07/27"
