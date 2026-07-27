"""Fast end-to-end sanity check of the complete DP+FL+HE stack.

Exercises every major module before launching the long drivers:
  1. Data loading and partitioning on both datasets.
  2. Model flatten/unflatten round-trip exactness.
  3. DP-SGD sigma solving and cumulative epsilon accounting.
  4. CKKS aggregation exactness (full HE and SPE) vs plaintext.
  5. Weighted homomorphic aggregation.
  6. Full orchestrator stack run with both DP+SPE and plain.

Prints PASS/FAIL for each check and exits nonzero if any fail.
Run from trial root: python run_smoke.py
"""
import sys
import numpy as np
import torch

from obscura import (config, data, models, dp, he, sensitivity, orchestrator,
                     aggregate, metrics)


def check_data_loading():
    """Load both datasets and assert expected shapes and partition coverage."""
    print("\n[1/7] Data loading...")
    checks_passed = 0

    X_str, y_str, _ = data.load_stroke()
    assert X_str.shape == (4908, 15), f"stroke shape mismatch: {X_str.shape}"
    checks_passed += 1

    X_loan, y_loan, _ = data.load_loan()
    assert X_loan.shape == (4269, 11), f"loan shape mismatch: {X_loan.shape}"
    checks_passed += 1

    for name in ["stroke", "loan"]:
        X, y, _ = data.load_dataset(name)
        X_tr, y_tr, X_te, y_te, info = data.split_and_scale(name, seed=config.SEED)
        n_train = info["n_train"]
        parts, pinfo = data.partition(name, y_tr, seed=config.SEED)

        all_indices = set()
        for part_indices in parts:
            all_indices.update(part_indices)
        assert len(all_indices) == n_train, \
            f"{name}: partition mismatch (covered {len(all_indices)}/{n_train})"
        checks_passed += 1

    print("  PASS: data shapes and partition coverage")
    return True


def check_flatten_unflatten():
    """Assert flatten/unflatten round-trip exactness."""
    print("[2/7] Flatten/unflatten round-trip...")
    torch.manual_seed(config.SEED)

    model = models.new_model(15, dataset="stroke")
    state1 = models.get_weights(model)

    flat = models.flatten_state(state1)
    assert isinstance(flat, np.ndarray), "flatten should return numpy array"
    assert flat.ndim == 1, "flatten should return 1-D array"

    state2 = models.unflatten_state(flat, state1)
    flat2 = models.flatten_state(state2)

    max_err = float(np.max(np.abs(flat - flat2)))
    assert max_err < 1e-10, f"flatten round-trip error: {max_err}"

    print(f"  PASS: flatten/unflatten round-trip (max error: {max_err:.2e})")
    return True


def check_dp_accounting():
    """Assert DP-SGD sigma solving and epsilon accounting."""
    print("[3/7] DP-SGD accounting...")
    torch.manual_seed(config.SEED)

    n_local = 1000
    target_eps = 9.0

    sigma = dp.get_sigma(target_eps, n_local, epochs=config.LOCAL_EPOCHS,
                         batch_size=config.BATCH_SIZE)
    actual_eps = dp.per_round_epsilon(sigma, n_local, epochs=config.LOCAL_EPOCHS,
                                      batch_size=config.BATCH_SIZE)

    eps_error = abs(actual_eps - target_eps)
    assert eps_error < 0.05, f"epsilon error {eps_error} >= 0.05"

    cum_eps = dp.cumulative_epsilon(sigma, n_local, rounds=10,
                                     epochs=config.LOCAL_EPOCHS,
                                     batch_size=config.BATCH_SIZE)
    assert cum_eps > actual_eps, \
        f"cumulative epsilon {cum_eps} not > per-round {actual_eps}"

    print(f"  PASS: DP accounting (target ε={target_eps}, actual {actual_eps:.3f}, "
          f"cumulative {cum_eps:.3f})")
    return True


def check_ckks_exactness():
    """CKKS aggregation exactness: full HE and SPE vs plaintext."""
    print("[4/7] CKKS aggregation exactness...")
    torch.manual_seed(config.SEED)

    model = models.new_model(15, dataset="stroke")
    state_template = models.state_template(15, dataset="stroke")
    ctx = he.make_context()

    base_state = models.get_weights(model)
    base_flat = models.flatten_state(base_state)

    n_clients = 5
    perturbed = [base_flat + 0.01 * np.random.randn(base_flat.shape[0])
                 for _ in range(n_clients)]

    plain_mean = aggregate.fedavg_weights(
        [1] * n_clients, weighted=False
    )
    plain_agg = np.mean(perturbed, axis=0)

    mask_full = np.ones(len(base_flat), dtype=bool)
    he_full_agg, _ = he.fedavg_he_timed(perturbed, mask_full, ctx, weights=None)

    mask_spe = sensitivity.random_mask(len(base_flat), config.SPE_DEFAULT, seed=config.SEED)
    spe_agg, _ = he.fedavg_he_timed(perturbed, mask_spe, ctx, weights=None)

    he_full_err = float(np.max(np.abs(he_full_agg - plain_agg)))
    spe_err = float(np.max(np.abs(spe_agg - plain_agg)))

    assert he_full_err < config.CKKS_TOLERANCE, \
        f"full HE error {he_full_err} >= tolerance"
    assert spe_err < config.CKKS_TOLERANCE, \
        f"SPE error {spe_err} >= tolerance"

    # Regression: fedavg_he_timed used to ignore its own `slots` argument --
    # encrypt_vector chunked at config.CKKS_SLOTS while the aggregation loop
    # chunked at `slots`, so any non-default slot count walked off the end of
    # the ciphertext list. fedavg_he and fedavg_he_timed are documented as the
    # same computation, so assert they agree at a slot count that forces
    # multiple chunks on this 3137-parameter model.
    small_slots = 1024
    he_small = he.fedavg_he(perturbed, mask_full, ctx, weights=None, slots=small_slots)
    he_small_timed, small_stats = he.fedavg_he_timed(
        perturbed, mask_full, ctx, weights=None, slots=small_slots)
    slots_err = float(np.max(np.abs(he_small_timed - plain_agg)))
    pair_err = float(np.max(np.abs(he_small_timed - he_small)))
    expected_chunks = -(-len(base_flat) // small_slots)
    assert slots_err < config.CKKS_TOLERANCE, \
        f"HE error at slots={small_slots} is {slots_err} >= tolerance"
    assert pair_err < config.CKKS_TOLERANCE, \
        f"fedavg_he vs fedavg_he_timed disagree at slots={small_slots}: {pair_err}"
    assert small_stats["n_chunks"] == expected_chunks, \
        f"n_chunks {small_stats['n_chunks']} != expected {expected_chunks}"

    # Regression: wire_payload_ratio must count the unencrypted remainder the
    # client still transmits, so at p < 1 it exceeds payload_ratio, and at p = 1
    # the two coincide because there is no remainder.
    n_enc = int(mask_spe.sum())
    _, spe_stats = he.fedavg_he_timed(perturbed, mask_spe, ctx, weights=None)
    expected_wire = ((spe_stats["ciphertext_bytes"] + 4 * (len(base_flat) - n_enc))
                     / float(4 * len(base_flat)))
    assert abs(spe_stats["wire_payload_ratio"] - expected_wire) < 1e-9, \
        f"wire_payload_ratio {spe_stats['wire_payload_ratio']} != {expected_wire}"
    assert spe_stats["wire_payload_ratio"] > spe_stats["payload_ratio"], \
        "wire_payload_ratio must exceed payload_ratio at p < 1"
    _, full_stats = he.fedavg_he_timed(perturbed, mask_full, ctx, weights=None)
    assert abs(full_stats["wire_payload_ratio"] - full_stats["payload_ratio"]) < 1e-9, \
        "wire_payload_ratio must equal payload_ratio at p = 1"

    print(f"  PASS: full-HE vs plaintext {he_full_err:.2e}, "
          f"SPE vs plaintext {spe_err:.2e}, "
          f"slots={small_slots} ({expected_chunks} chunks) {slots_err:.2e}, "
          f"wire ratio {spe_stats['wire_payload_ratio']:.3f} > "
          f"payload ratio {spe_stats['payload_ratio']:.3f}")
    return True


def check_weighted_he():
    """Weighted homomorphic aggregation."""
    print("[5/7] Weighted HE aggregation...")
    torch.manual_seed(config.SEED)

    ctx = he.make_context()

    n_clients = 5
    n_params = 500
    client_vecs = [np.random.randn(n_params).astype(np.float64) for _ in range(n_clients)]
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    plain_weighted = np.average(client_vecs, axis=0, weights=weights)

    mask_full = np.ones(n_params, dtype=bool)
    he_weighted, _ = he.fedavg_he_timed(client_vecs, mask_full, ctx, weights=weights)

    max_err = float(np.max(np.abs(he_weighted - plain_weighted)))
    assert max_err < config.CKKS_TOLERANCE, \
        f"weighted HE error {max_err} >= tolerance"

    print(f"  PASS: weighted HE vs weighted plaintext (max error: {max_err:.2e})")
    return True


def check_orchestrator_dp_spe():
    """Full stack run with DP+SPE."""
    print("[6/7] Orchestrator run (stack=dp_spe)...")
    torch.manual_seed(config.SEED)

    result = orchestrator.run_stack(
        dataset="stroke",
        stack="dp_spe",
        algo="fedprox",
        epsilon=config.EPSILON_DEFAULT,
        spe_fraction=config.SPE_DEFAULT,
        seed=config.SEED,
        rounds=2,
        local_epochs=1,
        log=lambda msg: None
    )

    required_keys = {"dataset", "stack", "algo", "seed", "epsilon", "spe_fraction",
                     "mask_mode", "rounds", "local_epochs", "width_scale", "n_params",
                     "use_smote", "metrics", "curve", "dp", "crypto", "timing",
                     "lossless", "mask"}
    assert set(result.keys()) == required_keys, \
        f"result keys mismatch: missing {required_keys - set(result.keys())}"

    assert result["lossless"]["max_abs_dev"] < config.CKKS_TOLERANCE, \
        f"lossless error {result['lossless']['max_abs_dev']} >= tolerance"

    encrypted_frac = result["crypto"].get("encrypted_fraction", 0.0)
    assert abs(encrypted_frac - config.SPE_DEFAULT) < 0.02, \
        f"encrypted_fraction {encrypted_frac} not within 0.02 of {config.SPE_DEFAULT}"

    assert result["dp"]["sigma"] is not None and result["dp"]["sigma"] > 0, \
        "DP sigma should be positive"

    print(f"  PASS: dp_spe stack run complete (lossless error "
          f"{result['lossless']['max_abs_dev']:.2e}, "
          f"encrypted_fraction {encrypted_frac:.3f})")
    return True


def check_orchestrator_plain():
    """Full stack run with plain (no DP, no HE)."""
    print("[7/7] Orchestrator run (stack=plain)...")
    torch.manual_seed(config.SEED)

    result = orchestrator.run_stack(
        dataset="stroke",
        stack="plain",
        algo="fedprox",
        seed=config.SEED,
        rounds=2,
        local_epochs=1,
        log=lambda msg: None
    )

    # The plaintext arm does no crypto, so its ONLY deviation from the float64
    # reference is the float32 quantisation of the stored state_dict -- i.e. it
    # must sit exactly ON the float32 floor, not at zero. That equality is the
    # calibration that makes the HE arms' "3-4x the floor" number meaningful.
    assert result["lossless"]["max_abs_dev"] == result["lossless"]["float32_floor"], \
        (f"plain deviation {result['lossless']['max_abs_dev']} should equal the "
         f"float32 floor {result['lossless']['float32_floor']}")

    assert result["crypto"].get("n_encrypted", 0) == 0, \
        f"plain should have no encrypted params, got {result['crypto'].get('n_encrypted')}"

    # The plaintext arm encrypts nothing, so its ciphertext payload_ratio is 0 --
    # but it still puts the entire model on the wire as float32, so its WIRE
    # ratio is exactly 1.0. Asserting both keeps the two payload metrics from
    # being conflated, and pins the plaintext end of the p axis the SPE rows are
    # measured against.
    assert result["crypto"].get("payload_ratio", None) == 0.0, \
        f"plain payload_ratio should be 0.0, got {result['crypto'].get('payload_ratio')}"
    assert result["crypto"].get("wire_payload_ratio", None) == 1.0, \
        (f"plain wire_payload_ratio should be 1.0, got "
         f"{result['crypto'].get('wire_payload_ratio')}")

    print(f"  PASS: plain stack run complete (no lossless error, no encryption, "
          f"wire_payload_ratio {result['crypto']['wire_payload_ratio']:.1f})")
    return True


def main():
    """Run all smoke checks."""
    print("=" * 60)
    print("Trial 07/27 — END-TO-END SMOKE TEST")
    print("=" * 60)

    checks = [
        check_data_loading,
        check_flatten_unflatten,
        check_dp_accounting,
        check_ckks_exactness,
        check_weighted_he,
        check_orchestrator_dp_spe,
        check_orchestrator_plain,
    ]

    failed = []
    for check_fn in checks:
        try:
            check_fn()
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}")
            failed.append((check_fn.__name__, str(e)))

    print("\n" + "=" * 60)
    if not failed:
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print(f"✗ {len(failed)} check(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error[:80]}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
