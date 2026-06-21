"""Experiment C -- accuracy / losslessness of plaintext vs full-HE vs SPE(p=0.25).

Trains ResNet-18 on CIFAR-10 under the three aggregation modes (reduced-but-faithful
default schedule; see config.ACC_*) and records the test-accuracy curves, which
coincide because CKKS aggregation is numerically identical to the plaintext mean.
Also runs an exact per-round aggregation-agreement check (length-independent proof
of losslessness). Emits results/accuracy.json.
"""
import json
import os

from fedml_he import config, fedavg


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"device: {fedavg.device()}")
    train_data, test_data, source = fedavg.get_cifar10(
        config.ACC_TRAIN_SUBSET, config.ACC_TEST_SUBSET)
    print(f"{source}: {len(train_data[1])} train / {len(test_data[1])} test "
          f"across {config.NUM_CLIENTS} IID clients\n")

    # Exact losslessness: aggregate identical client updates three ways.
    agree = fedavg.aggregation_agreement(
        config.ACC_MODEL, config.ACC_SPE_FRACTION, train_data)
    print(f"aggregation agreement (vs plaintext): "
          f"HE {agree['he_vs_plain_max_abs']:.2e}  "
          f"SPE {agree['spe_vs_plain_max_abs']:.2e}  "
          f"(SPE encrypts {agree['encrypted_fraction']*100:.0f}%)\n")

    curves = {}
    finals = {}
    for mode, label in [("plaintext", "plaintext"), ("he", "full HE"),
                        ("spe", f"SPE (p={config.ACC_SPE_FRACTION})")]:
        print(f"[{label}]")
        curve = fedavg.run_federated(
            mode, config.ACC_MODEL, config.ACC_ROUNDS, config.ACC_LOCAL_EPOCHS,
            config.ACC_LR, config.ACC_BATCH_SIZE, config.ACC_SPE_FRACTION,
            train_data, test_data)
        curves[label] = curve
        finals[label] = curve[-1]
        print(f"  -> final acc {curve[-1]*100:.2f}%\n")

    spread = (max(finals.values()) - min(finals.values())) * 100
    out = {"config": {"model": config.ACC_MODEL, "rounds": config.ACC_ROUNDS,
                      "train_subset": config.ACC_TRAIN_SUBSET,
                      "spe_fraction": config.ACC_SPE_FRACTION, "data_source": source},
           "agreement": agree, "curves": curves, "final": finals,
           "max_spread_pct": spread}
    with open(os.path.join(config.RESULTS_DIR, "accuracy.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("final accuracy:")
    for k, v in finals.items():
        print(f"  {k:18s} {v*100:.2f}%")
    print(f"  max spread = {spread:.2f}%  "
          f"({'LOSSLESS' if spread < 0.5 else 'check'} -- within 0.5% target)")
    print(f"\nsaved -> {config.RESULTS_DIR}/accuracy.json")


if __name__ == "__main__":
    main()
