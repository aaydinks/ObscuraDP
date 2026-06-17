# ObscuraDP
ObscuraDP is an innovative differential privacy framework built to protect sensitive user data from AI misuse. It incorporates 

## Experiments

- [**Experiment 1**](./Experiment%201) — Reproduction of **SMOTETomek + FedProx**
  ([arXiv:2508.10017](https://arxiv.org/abs/2508.10017)): the ε-vs-recall frontier
  on non-IID clinical (stroke) data under DP-SGD. Self-contained — code, dataset,
  results, figures, and write-up live in the [`Experiment 1/`](./Experiment%201)
  folder (see its [README](./Experiment%201/README.md)).

- [**Experiment 2**](./Experiment%202) — Reproduction of the **Standard FL vs.
  DP-FL vs. HE-FL** three-way comparison on credit-risk loan approval
  ([Sci. Rep. 10.1038/s41598-025-34536-9](https://doi.org/10.1038/s41598-025-34536-9)):
  vanilla FedAvg vs. Opacus DP-SGD vs. TenSEAL CKKS homomorphic aggregation,
  quantifying DP's accuracy cost and HE's accuracy-neutrality (overhead without
  accuracy loss). Self-contained in [`Experiment 2/`](./Experiment%202)
  (see its [README](./Experiment%202/README.md)).
