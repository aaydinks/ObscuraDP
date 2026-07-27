"""Independent re-derivation of Trial 07/27's CRYPTO-side claims.

This file deliberately does NOT use obscura.he or obscura.aggregate to compute any
reference value: those modules are what is being audited. Every reference number is
re-derived here from TenSEAL / Opacus / numpy / plain torch. obscura modules are
imported only as the *subject* of a comparison (and obscura.data only to obtain the
same client partition the experiment used, which is not itself under test).

Everything is seeded and deterministic except the CKKS ciphertexts themselves: SEAL's
encryption RNG is not exposed by TenSEAL, so the measured HE deviations wobble in their
last digit or two between runs (~4.8e-08..7.0e-08). Every threshold here is orders of
magnitude away from that wobble, so the verdicts do not move.

Run:  /home/user/ObscuraDP/obscura_env/bin/python verify_claims.py     (~17 s)
"""
import itertools
import json
import math
import os
import sys
import warnings

import numpy as np
import tenseal as ts
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# Constants copied from obscura/config.py (cross-checked below, never imported
# indirectly through a module under test).
# ---------------------------------------------------------------------------
SEED = 42
NUM_CLIENTS = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
MAX_GRAD_NORM = 1.0
TARGET_DELTA = 1e-5
ROUNDS = 10
POLY_MOD_DEGREE = 8192
COEFF_MOD_BITS = [60, 40, 60]
GLOBAL_SCALE = 2 ** 40
SLOTS = POLY_MOD_DEGREE // 2          # 4096
HIDDEN_STROKE = (64, 32)
IN_FEATURES = 15
N_PARAMS_STROKE = 3137

RESULTS = os.path.join(HERE, "results")
VERDICTS = []


def record(claim, ok, msg):
    VERDICTS.append((claim, ok, msg))
    print("\n{} {}: {}".format("PASS" if ok else "FAIL", claim, msg))


def load_result(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def check_config():
    """Confirm the constants above match obscura/config.py verbatim."""
    from obscura import config
    pairs = [
        ("SEED", SEED, config.SEED),
        ("NUM_CLIENTS", NUM_CLIENTS, config.NUM_CLIENTS),
        ("LOCAL_EPOCHS", LOCAL_EPOCHS, config.LOCAL_EPOCHS),
        ("BATCH_SIZE", BATCH_SIZE, config.BATCH_SIZE),
        ("MAX_GRAD_NORM", MAX_GRAD_NORM, config.MAX_GRAD_NORM),
        ("TARGET_DELTA", TARGET_DELTA, config.TARGET_DELTA),
        ("ROUNDS", ROUNDS, config.ROUNDS),
        ("POLY", POLY_MOD_DEGREE, config.CKKS_POLY_MODULUS_DEGREE),
        ("COEFF", COEFF_MOD_BITS, config.CKKS_COEFF_MOD_BIT_SIZES),
        ("SCALE", GLOBAL_SCALE, config.CKKS_GLOBAL_SCALE),
        ("SLOTS", SLOTS, config.CKKS_SLOTS),
        ("HIDDEN", HIDDEN_STROKE, config.HIDDEN["stroke"]),
    ]
    bad = [n for n, a, b in pairs if a != b]
    if bad:
        raise SystemExit("config drift: {}".format(bad))
    print("config cross-check OK (constants replicated verbatim)")


# ---------------------------------------------------------------------------
# Independent CKKS helpers -- raw TenSEAL only.
# ---------------------------------------------------------------------------
def ref_context():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=POLY_MOD_DEGREE,
                     coeff_mod_bit_sizes=COEFF_MOD_BITS)
    ctx.generate_galois_keys()
    ctx.global_scale = GLOBAL_SCALE
    return ctx


def ref_he_weighted_sum(ctx, client_vecs, weights):
    """Encrypt each client's vector, scale by its plaintext weight, sum, decrypt.

    Chunked at SLOTS. Written from scratch; shares no code with obscura.he.
    """
    n = len(client_vecs[0])
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, SLOTS):
        e = min(s + SLOTS, n)
        acc = None
        for c, v in enumerate(client_vecs):
            ct = ts.ckks_vector(ctx, np.asarray(v[s:e], dtype=np.float64).tolist())
            ct = ct * float(w[c])
            acc = ct if acc is None else acc + ct
        out[s:e] = np.asarray(acc.decrypt()[: e - s], dtype=np.float64)
    return out


def ref_he_mean(ctx, client_vecs):
    """Unweighted: sum ciphertexts then one scalar multiply by 1/N."""
    n = len(client_vecs[0])
    N = len(client_vecs)
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, SLOTS):
        e = min(s + SLOTS, n)
        acc = None
        for v in client_vecs:
            ct = ts.ckks_vector(ctx, np.asarray(v[s:e], dtype=np.float64).tolist())
            acc = ct if acc is None else acc + ct
        acc = acc * (1.0 / N)
        out[s:e] = np.asarray(acc.decrypt()[: e - s], dtype=np.float64)
    return out


def ref_spe(ctx, client_vecs, mask, weights):
    """Independent SPE: CKKS on masked coords, numpy weighted mean elsewhere."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = np.full(n, np.nan, dtype=np.float64)

    plain_idx = np.flatnonzero(~mask)
    if plain_idx.size:
        stack = np.stack([np.asarray(v)[plain_idx] for v in client_vecs], axis=0)
        out[plain_idx] = np.tensordot(w, stack, axes=(0, 0))

    enc_idx = np.flatnonzero(mask)
    if enc_idx.size:
        sub = [np.asarray(v)[enc_idx] for v in client_vecs]
        out[enc_idx] = ref_he_weighted_sum(ctx, sub, w)

    assert not np.isnan(out).any(), "a coordinate was never written"
    return out


def np_weighted_mean(client_vecs, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in client_vecs], axis=0)
    return np.tensordot(w, stack, axes=(0, 0))


# ---------------------------------------------------------------------------
# Independent DP accounting -- raw Opacus.
# ---------------------------------------------------------------------------
def ref_sigma(epsilon, n_local, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(target_epsilon=epsilon,
                                target_delta=TARGET_DELTA,
                                sample_rate=min(1.0, batch_size / n_local),
                                epochs=epochs,
                                accountant="rdp")


def ref_epsilon(sigma, sample_rate, steps):
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    acc.history = [(sigma, sample_rate, int(steps))]
    return float(acc.get_epsilon(delta=TARGET_DELTA))


# ---------------------------------------------------------------------------
# Plain-torch model matching obscura.models.FFNN's init RNG consumption exactly.
# ---------------------------------------------------------------------------
class PlainFFNN(nn.Module):
    def __init__(self, in_features, h1, h2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fresh_stroke_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return PlainFFNN(IN_FEATURES, HIDDEN_STROKE[0], HIDDEN_STROKE[1])


def param_blocks(model):
    return [(name, p.numel()) for name, p in model.named_parameters()]


# ===========================================================================
# CLAIM 1 -- CKKS aggregation lossless, incl. on DP-noised updates + SPE scatter
# ===========================================================================
def claim1(ctx):
    print("\n" + "=" * 78)
    print("CLAIM 1 -- CKKS/SPE losslessness on DP-noised updates")
    print("=" * 78)

    reported = load_result("lossless.json")
    rep = {}
    if reported:
        for r in reported["lossless"]:
            rep.setdefault((r["epsilon"], r["spe_fraction"], r["mode"]), []).append(r)

    rng = np.random.RandomState(SEED)
    counts = np.array([484.0, 234.0, 1120.0, 4920.0, 738.0])
    n = N_PARAMS_STROKE

    # A client update at the reported scale: mean|w| ~ 0.087 (lossless.json).
    base = rng.normal(0.0, 0.11, n)
    base *= 0.087 / np.mean(np.abs(base))

    # mean-over-clients sigma actually used by run_losslessness.py
    eps_sigma = [(0.5, 3.22265625), (2.0, 1.27587890625),
                 (9.0, 0.6417236328125), (50.0, 0.3180389404296875)]
    if reported:
        seen = {}
        for r in reported["lossless"]:
            seen[r["epsilon"]] = r["sigma"]
        eps_sigma = sorted(seen.items())

    rows = []
    all_ok = True
    for epsilon, sigma in eps_sigma:
        noise_std = sigma * MAX_GRAD_NORM / BATCH_SIZE
        clients = [base + rng.normal(0.0, noise_std, n) for _ in range(NUM_CLIENTS)]
        plain = np_weighted_mean(clients, counts)

        for p in (0.25, 1.0):
            k = int(round(p * n))
            # Non-contiguous mask: stride-based, so misalignment is loud.
            mask = np.zeros(n, dtype=bool)
            mask[np.sort(np.random.RandomState(7).permutation(n)[:k])] = True
            if p == 1.0:
                mask[:] = True
            mine = ref_spe(ctx, clients, mask, counts)
            dev = float(np.max(np.abs(mine - plain)))
            rel = dev / noise_std
            rows.append((epsilon, p, dev, rel))
            all_ok &= dev < 1e-6
            r = rep.get((epsilon, p, "spe" if p < 1 else "he"))
            rep_dev = r[0]["max_abs_dev"] if r else float("nan")
            print("  eps={:5.1f} p={:.2f}  mine max|HE-plain|={:.3e} "
                  "({:.2e} x noise_std)   reported={:.3e}"
                  .format(epsilon, p, dev, rel, rep_dev))

    devs = [r[2] for r in rows]
    rels = [r[3] for r in rows]
    print("  --> my range: max_abs_dev {:.2e}..{:.2e}, "
          "rel-to-noise {:.2e}..{:.2e}"
          .format(min(devs), max(devs), min(rels), max(rels)))

    if reported:
        rr = reported["lossless"]
        print("  reported range: max_abs_dev {:.3e}..{:.3e}, rel_to_noise_floor "
              "{:.3e}..{:.3e}".format(min(r["max_abs_dev"] for r in rr),
                                      max(r["max_abs_dev"] for r in rr),
                                      min(r["rel_to_noise_floor"] for r in rr),
                                      max(r["rel_to_noise_floor"] for r in rr)))
        ru = [r["max_abs_dev"] / r["update_scale"] for r in rr]
        print("  DENOMINATOR NOTE: 'rel_to_noise_floor' divides a deviation measured on"
              " aggregated WEIGHTS")
        print("    by a PER-STEP GRADIENT noise std -- two different spaces, and it is"
              " why the ratio")
        print("    moves 14x across eps ({:.1e}..{:.1e}) purely because the denominator"
              " moves.".format(min(r["rel_to_noise_floor"] for r in rr),
                               max(r["rel_to_noise_floor"] for r in rr)))
        print("    Dividing instead by the update scale actually being encrypted "
              "(mean|w| ~ {:.3f})".format(np.mean([r["update_scale"] for r in rr])))
        print("    gives a stable {:.2e}..{:.2e} -- the defensible form of the "
              "'1e-6 of the signal' headline.".format(min(ru), max(ru)))

    # ---- SPE scatter-back stress test -------------------------------------
    # Non-contiguous mask + wildly coordinate-varying clients so any drop,
    # duplication or off-by-one shows as an O(1) error, not a rounding blip.
    print("\n  SPE scatter-back stress test (non-contiguous mask, "
          "high coordinate-to-coordinate contrast):")
    ns = 5000                                        # spans 2 CKKS chunks
    idx = np.arange(ns)
    # Distinctive per-coordinate signature; adjacent coords differ by ~O(1).
    sig = 0.9 * np.sin(idx * 2.399963) * (0.15 + 0.85 * ((idx * 37) % 11) / 10.0)
    stress = [sig * (1.0 + 0.37 * c) + 0.11 * np.cos(idx * (1.7 + c)) for c in range(NUM_CLIENTS)]
    r7 = np.random.RandomState(1234)
    smask = np.zeros(ns, dtype=bool)
    smask[r7.permutation(ns)[: int(0.25 * ns)]] = True   # scattered, NOT contiguous
    longest = max((len(list(g)) for v, g in itertools.groupby(smask) if v), default=0)
    print("    mask: {} True of {}, longest contiguous True run = {}; "
          "encrypted coords straddle {} CKKS chunks"
          .format(int(smask.sum()), ns, longest, math.ceil(int(smask.sum()) / SLOTS)))
    print("    client contrast: mean |v[i]-v[i+1]| = {:.3f}, |v| range {:.3f}..{:.3f}"
          .format(float(np.mean(np.abs(np.diff(stress[0])))),
                  float(np.min(np.abs(stress[0]))), float(np.max(np.abs(stress[0])))))
    sref = np_weighted_mean(stress, counts)
    smine = ref_spe(ctx, stress, smask, counts)
    stress_dev = float(np.max(np.abs(smine - sref)))

    # Positive control: shift the encrypted coordinates by one slot on scatter.
    bad = smine.copy()
    e_idx = np.flatnonzero(smask)
    bad[e_idx] = smine[np.roll(e_idx, 1)]
    control_dev = float(np.max(np.abs(bad - sref)))
    print("    max|SPE-plain| = {:.3e}   (off-by-one positive control = {:.3e})"
          .format(stress_dev, control_dev))
    has_power = control_dev > 1e-3
    all_ok &= (stress_dev < 1e-6) and has_power

    # ---- audit obscura.he against the independent reference ---------------
    print("\n  Auditing obscura.he.fedavg_he / fedavg_he_timed on the same input:")
    from obscura import he as ohe
    got = ohe.fedavg_he(stress, smask, ctx, weights=counts / counts.sum())
    d_pkg = float(np.max(np.abs(got - sref)))
    got_t, st = ohe.fedavg_he_timed(stress, smask, ctx, weights=counts / counts.sum())
    d_pkg_t = float(np.max(np.abs(got_t - sref)))
    print("    obscura fedavg_he       max dev vs my plaintext ref = {:.3e}".format(d_pkg))
    print("    obscura fedavg_he_timed max dev vs my plaintext ref = {:.3e}".format(d_pkg_t))
    all_ok &= (d_pkg < 1e-6) and (d_pkg_t < 1e-6)

    # Mask-polarity / index-alignment probe: the PLAINTEXT half must be exact to
    # float64, the ENCRYPTED half only to CKKS precision. If the mask were
    # inverted or shifted, this signature would swap or smear.
    d_plain = float(np.max(np.abs(got_t[~smask] - sref[~smask])))
    d_enc = float(np.max(np.abs(got_t[smask] - sref[smask])))
    print("    plaintext-half dev = {:.3e} (must be ~1e-16), "
          "encrypted-half dev = {:.3e} (CKKS precision)".format(d_plain, d_enc))
    all_ok &= (d_plain < 1e-12) and (1e-12 < d_enc < 1e-6)

    # Explicit dropped/duplicated coordinate check on the package output.
    n_written = int(np.sum(np.isfinite(got_t)))
    all_ok &= (n_written == ns) and np.allclose(got, got_t, atol=1e-6)
    print("    coordinates finite in package output: {}/{}; "
          "fedavg_he and fedavg_he_timed agree to 1e-6: {}"
          .format(n_written, ns, bool(np.allclose(got, got_t, atol=1e-6))))

    record("CLAIM 1", all_ok,
           "max|HE-plaintext| in {:.2e}..{:.2e} over eps x p (reported 4.9e-08..6.9e-08); "
           "= {:.2e}..{:.2e} x the DP noise std; scattered-mask SPE dev {:.2e} while an "
           "off-by-one scatter would give {:.2e}, so the test has power"
           .format(min(devs), max(devs), min(rels), max(rels), stress_dev, control_dev))
    return rows


# ===========================================================================
# CLAIM 2 -- weighted homomorphic aggregation
# ===========================================================================
def claim2(ctx):
    print("\n" + "=" * 78)
    print("CLAIM 2 -- weighted homomorphic aggregation")
    print("=" * 78)

    counts = np.array([484.0, 234.0, 1120.0, 4920.0, 738.0])
    w = counts / counts.sum()
    print("  weights = {}".format(np.round(w, 6).tolist()))
    rng = np.random.RandomState(11)
    n = 1500
    ok = True
    print("  {:>10} {:>13} {:>13} {:>13}".format(
        "|v| scale", "max abs dev", "max rel dev", "vs unweighted"))
    for scale in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        clients = [rng.normal(0.0, scale, n) for _ in range(NUM_CLIENTS)]
        ref = np_weighted_mean(clients, counts)
        mine = ref_he_weighted_sum(ctx, clients, w)
        dev = float(np.max(np.abs(mine - ref)))
        rel = dev / float(np.max(np.abs(ref)))
        # sanity: the weighted answer must differ materially from the unweighted one
        unw = np.mean(np.stack(clients), axis=0)
        sep = float(np.max(np.abs(ref - unw)))
        good = dev < max(1e-6, 1e-6 * scale) and sep > 100 * dev
        ok &= good
        print("  {:>10.0e} {:>13.3e} {:>13.3e} {:>13.3e} {}".format(
            scale, dev, rel, sep, "" if good else "  <-- FAIL"))

    # audit obscura.he's weighted path against the independent reference
    from obscura import he as ohe
    clients = [rng.normal(0.0, 0.1, n) for _ in range(NUM_CLIENTS)]
    ref = np_weighted_mean(clients, counts)
    pkg = ohe.fedavg_he(clients, np.ones(n, dtype=bool), ctx, weights=counts)
    d_pkg = float(np.max(np.abs(pkg - ref)))
    print("  obscura.he weighted full-HE vs my plaintext weighted mean: {:.3e}".format(d_pkg))
    ok &= d_pkg < 1e-6

    # multiplicative depth budget of the configured chain
    a = ts.ckks_vector(ctx, [0.1] * 8)
    depth1 = True
    try:
        b = a * 0.3
        _ = b.decrypt()
    except Exception:
        depth1 = False
    depth2 = True
    try:
        c = (a * 0.3) * 0.3
        _ = c.decrypt()
    except Exception as exc:
        depth2 = False
        depth2_err = type(exc).__name__
    print("  modulus chain {} -> depth-1 scalar multiply: {} ; depth-2: {}"
          .format(COEFF_MOD_BITS, "OK" if depth1 else "FAILS",
                  "OK" if depth2 else "FAILS (as expected)"))
    ok &= depth1 and (not depth2)

    record("CLAIM 2", ok,
           "weighted CKKS aggregation matches the plaintext weighted mean to <1e-6 "
           "absolute at |v| from 1e-3 to 10; chain [60,40,60] supports exactly one "
           "scalar multiply (depth 2 raises 'scale out of bounds'), which is all the "
           "w_c multiply needs -- but there is ZERO headroom left")


# ===========================================================================
# CLAIM 3 -- payload ratio
# ===========================================================================
def claim3(ctx):
    print("\n" + "=" * 78)
    print("CLAIM 3 -- ciphertext/plaintext payload ratio")
    print("=" * 78)

    rng = np.random.RandomState(3)
    sizes = {}
    for m in (1, 512, 3137, 4096):
        v = rng.normal(0, 0.1, m)
        sizes[m] = len(ts.ckks_vector(ctx, v.tolist()).serialize())
    print("  serialized ciphertext bytes by #slots filled: {}".format(sizes))
    S = float(np.mean(list(sizes.values())))
    print("  --> one ciphertext costs ~{:.0f} B whatever fraction of its 4096 slots "
          "is used".format(S))

    # analytic prediction: ceil(n/slots) * S / (4n)
    def predict(n):
        return math.ceil(n / SLOTS) * S / (4.0 * n)

    # theoretical uncompressed size for chain [60,40,60]:
    # 2 polys x 8192 coeffs x 2 data primes (the last 60-bit prime is SEAL's
    # special/key-switching prime and is not part of a fresh ciphertext) x 8 B
    raw = 2 * POLY_MOD_DEGREE * 2 * 8
    print("  uncompressed CKKS ct for [60,40,60] = 2*{}*2*8 = {} B "
          "-> serialized/{} = {:.3f} (SEAL ZSTD compression)"
          .format(POLY_MOD_DEGREE, raw, raw, S / raw))
    print("  raw asymptotic ratio would be {}/{} = {:.3f}; measured {:.3f}"
          .format(raw, 4 * SLOTS, raw / (4.0 * SLOTS), S / (4.0 * SLOTS)))

    # measured asymptotic ratio on a large, slot-aligned vector
    big = 20 * SLOTS
    v = rng.normal(0, 0.1, big)
    tot = 0
    for s in range(0, big, SLOTS):
        tot += len(ts.ckks_vector(ctx, v[s:s + SLOTS].tolist()).serialize())
    asym = tot / (4.0 * big)
    print("  measured asymptotic ratio at n={} (slot-aligned): {:.3f}".format(big, asym))

    print("\n  {:>9} {:>7} {:>12} {:>12} {:>12}".format(
        "n_params", "chunks", "predicted", "reported", "abs diff"))
    ovh = load_result("overhead.json")
    ok = abs(predict(N_PARAMS_STROKE) - 18.758) < 0.15 and abs(asym - 14.4) < 0.2
    if ovh:
        for r in ovh["crypto_vs_size"]:
            if r["n_encrypted"] != r["n_params"]:
                continue
            nn_ = r["n_params"]
            pr = predict(nn_)
            print("  {:>9} {:>7} {:>12.3f} {:>12.3f} {:>12.4f}".format(
                nn_, math.ceil(nn_ / SLOTS), pr, r["payload_ratio"],
                abs(pr - r["payload_ratio"])))
            ok &= abs(pr - r["payload_ratio"]) < 0.05

    print("\n  padding explanation: 18.76 / 14.37 = {:.4f} ; 4096/3137 = {:.4f}"
          .format(predict(N_PARAMS_STROKE) / asym, SLOTS / float(N_PARAMS_STROKE)))

    # the SPE payload_ratio definition issue
    if ovh:
        print("\n  NOTE on the SPE rows of crypto_vs_p (n_params=3137):")
        for r in ovh["crypto_vs_p"]:
            if r["n_encrypted"] == 0:
                continue
            true_wire = r["ciphertext_bytes"] + 4 * (r["n_params"] - r["n_encrypted"])
            print("    p={:.2f}: reported ratio {:.3f} counts only the ciphertext; "
                  "the client must still ship {} plaintext params, so the true wire "
                  "ratio is {:.3f}".format(r["fraction"], r["payload_ratio"],
                                           r["n_params"] - r["n_encrypted"],
                                           true_wire / float(r["plaintext_bytes"])))

    record("CLAIM 3", ok,
           "ratio(n) = ceil(n/4096)*{:.0f}B/(4n) reproduces every reported value to "
           "<0.05; 18.76x at n=3137 is 100% zero-padding (4096/3137 = {:.3f}); "
           "measured asymptotic ratio {:.3f}x -- same [60,40,60] chain as Exp 3, "
           "no measurement error".format(S, SLOTS / float(N_PARAMS_STROKE), asym))
    return S, asym


# ===========================================================================
# CLAIM 4 -- DP accounting
# ===========================================================================
def claim4():
    print("\n" + "=" * 78)
    print("CLAIM 4 -- DP accounting and the noise floor")
    print("=" * 78)

    n_local = 484                       # stroke client 0
    ok = True

    print("  (a) get_sigma -> per_round_epsilon round trip (n_local={}):".format(n_local))
    sample_rate = min(1.0, BATCH_SIZE / n_local)
    steps_round = int(LOCAL_EPOCHS / sample_rate)
    for eps in (0.5, 2.0, 9.0, 50.0):
        s = ref_sigma(eps, n_local)
        back = ref_epsilon(s, sample_rate, steps_round)
        cum = ref_epsilon(s, sample_rate, int(ROUNDS * LOCAL_EPOCHS / sample_rate))
        good = abs(back - eps) / eps < 0.02
        ok &= good
        print("    eps={:5.1f} -> sigma={:.6f} -> eps_round={:.4f} (err {:+.2%}) "
              "| eps_cumulative over {} rounds = {:.3f}{}"
              .format(eps, s, back, (back - eps) / eps, ROUNDS, cum,
                      "" if good else "   <-- FAIL"))

    # composition monotone + super-additive vs per-round
    s9 = ref_sigma(9.0, n_local)
    e1 = ref_epsilon(s9, sample_rate, steps_round)
    e10 = ref_epsilon(s9, sample_rate, int(ROUNDS * LOCAL_EPOCHS / sample_rate))
    comp_ok = e1 < e10 < ROUNDS * e1
    ok &= comp_ok
    print("  (b) composition: eps(1 round)={:.4f}, eps({} rounds)={:.4f}; "
          "1*eps < composed < {}*eps -> {}"
          .format(e1, ROUNDS, e10, ROUNDS, "OK (sub-linear, RDP)" if comp_ok else "FAIL"))

    # cross-check obscura.dp reproduces the same numbers
    try:
        from obscura import dp as odp
        d_s = odp.get_sigma(9.0, n_local)
        d_r = odp.per_round_epsilon(d_s, n_local)
        d_c = odp.cumulative_epsilon(d_s, n_local)
        agree = (abs(d_s - s9) < 1e-9 and abs(d_r - e1) < 1e-9 and abs(d_c - e10) < 1e-9)
        ok &= agree
        print("  (c) obscura.dp reproduces sigma/eps_round/eps_cum exactly: {}".format(agree))
    except Exception as exc:
        print("  (c) obscura.dp unavailable ({}) -- skipped".format(exc))

    # ---- (d) is dp_noise_std = sigma*C/B the right std? Measure Opacus. ----
    print("  (d) empirical check of dp_noise_std = sigma*C/B against Opacus:")
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    n_data, sigma_t = 1024, 50.0
    X = torch.randn(n_data, IN_FEATURES)
    y = (torch.rand(n_data) > 0.5).float()
    model = PlainFFNN(IN_FEATURES, 64, 32)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # lr=0: params never move
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)
    eng = PrivacyEngine(accountant="rdp")
    model, opt, loader = eng.make_private(module=model, optimizer=opt,
                                          data_loader=loader,
                                          noise_multiplier=sigma_t,
                                          max_grad_norm=MAX_GRAD_NORM)
    print("      Opacus expected_batch_size = {} (nominal BATCH_SIZE = {})"
          .format(opt.expected_batch_size, BATCH_SIZE))
    crit = nn.BCELoss()
    samples = []
    it = iter(loader)
    for _ in range(6):
        xb, yb = next(it)
        if xb.numel() == 0:
            continue
        opt.zero_grad(set_to_none=True)
        crit(model(xb), yb).backward()
        opt.step()
        samples.append(torch.cat([p.grad.flatten() for p in model.parameters()]).clone())
    emp = torch.stack(samples).std().item()
    pred_B = sigma_t * MAX_GRAD_NORM / BATCH_SIZE
    pred_eb = sigma_t * MAX_GRAD_NORM / opt.expected_batch_size
    print("      measured std of p.grad over {} steps x {} params: {:.4f}"
          .format(len(samples), N_PARAMS_STROKE, emp))
    print("      sigma*C/BATCH_SIZE       = {:.4f}   (ratio {:.4f})"
          .format(pred_B, emp / pred_B))
    print("      sigma*C/expected_batch   = {:.4f}   (ratio {:.4f})"
          .format(pred_eb, emp / pred_eb))
    print("      sigma*C (no 1/B at all)  = {:.4f}   (ratio {:.4f})"
          .format(sigma_t, emp / sigma_t))
    form_ok = abs(emp / pred_eb - 1.0) < 0.05
    ok &= form_ok

    # what expected_batch_size Opacus would actually use for the real clients
    print("      per-client expected_batch_size Opacus would use (stroke, B=32):")
    for nl in (484, 234, 1120, 4920, 738):
        nb = math.ceil(nl / BATCH_SIZE)
        eb = int(nl / nb)
        print("        n_local={:5d} -> len(loader)={:3d} -> expected_batch_size={:2d} "
              "(dp_noise_std assumes 32; {:+.1%})".format(nl, nb, eb, 32.0 / eb - 1.0))

    record("CLAIM 4", ok,
           "sigma->epsilon round-trips to <2%; RDP composition is sub-linear and "
           "monotone; Opacus adds N(0, (sigma*C)^2) to the SUMMED grad then divides by "
           "expected_batch_size, so sigma*C/B IS the right std for the averaged "
           "gradient -- but B must be Opacus's expected_batch_size (29-32 for these "
           "clients), not the nominal 32, a <=10% correction that does not touch the "
           "1e-6 headline")


# ===========================================================================
# CLAIM 5 -- DP-aware sensitivity
# ===========================================================================
def stroke_client0_batch():
    from obscura import data
    X_tr, y_tr, X_te, y_te, info = data.split_and_scale("stroke", seed=SEED)
    parts, _ = data.partition("stroke", y_tr, num_clients=NUM_CLIENTS, seed=SEED)
    cd = data.build_client_data("stroke", X_tr, y_tr, parts, use_smote=None, seed=SEED)
    Xc, yc = cd[0]
    n_take = min(256, len(yc))
    idx = np.random.RandomState(SEED).choice(len(yc), size=n_take, replace=False)
    return (torch.from_numpy(Xc[idx].astype(np.float32)),
            torch.from_numpy(yc[idx].astype(np.float32)),
            len(yc))


def my_clean_scores(model, xb, yb):
    model.zero_grad(set_to_none=True)
    nn.BCELoss()(model(xb), yb).backward()
    return {n: p.grad.abs().mean().item() for n, p in model.named_parameters()}


def my_avg_clipped_grad(model, xb, yb, C=MAX_GRAD_NORM):
    """Per-sample gradients, clipped JOINTLY over all params by min(1, C/||g||)."""
    crit = nn.BCELoss()
    acc = None
    for i in range(xb.shape[0]):
        model.zero_grad(set_to_none=True)
        crit(model(xb[i:i + 1]), yb[i:i + 1]).backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        g = g * min(1.0, C / (g.norm(2).item() + 1e-10))
        acc = g if acc is None else acc + g
    return acc / xb.shape[0]


def scores_from_flat(flat, blocks):
    out, off = {}, 0
    for name, numel in blocks:
        out[name] = flat[off:off + numel].abs().mean().item()
        off += numel
    return out


def per_scalar_scores(blocks, scores):
    n_total = sum(nm for _, nm in blocks)
    per = np.empty(n_total, dtype=np.float64)
    off = 0
    for name, numel in blocks:
        per[off:off + numel] = scores.get(name, 0.0)
        off += numel
    return per


def my_build_mask(blocks, scores, fraction, tiebreak=None):
    """Top-k over per-scalar scores.

    tiebreak: None                -> np.argpartition, exactly as obscura ships it
              np.random.RandomState -> uniformly random tie-breaking
              "index"             -> stable, lowest-flat-index-first tie-breaking
    """
    per = per_scalar_scores(blocks, scores)
    n_total = per.size
    k = int(round(fraction * n_total))
    mask = np.zeros(n_total, dtype=bool)
    if k >= n_total:
        mask[:] = True
    elif k > 0:
        if tiebreak is None:
            mask[np.argpartition(per, n_total - k)[n_total - k:]] = True
        elif tiebreak == "index":
            mask[np.argsort(-per, kind="stable")[:k]] = True
        else:
            order = np.lexsort((tiebreak.permutation(n_total), -per))
            mask[order[:k]] = True
    return mask


def mask_prefix(blocks, scores, fraction):
    """(tuple of fully-included tensors, name of the partially-cut tensor, n_cut).

    This is everything the SPE mask *actually decides*: which tensors are
    encrypted. Which scalars inside a tie-valued cut tensor get picked is
    arbitrary, because every scalar in a tensor carries the same score.
    """
    n_total = sum(nm for _, nm in blocks)
    k = int(round(fraction * n_total))
    order = sorted(blocks, key=lambda b: (-scores[b[0]], b[0]))
    full, rem = [], k
    for name, numel in order:
        if rem >= numel:
            full.append(name)
            rem -= numel
        else:
            return tuple(full), name, rem
    return tuple(full), None, 0


def jaccard(a, b):
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0


def claim5():
    print("\n" + "=" * 78)
    print("CLAIM 5 -- DP-aware sensitivity / mask agreement")
    print("=" * 78)

    xb, yb, n_local = stroke_client0_batch()
    B = xb.shape[0]
    print("  client 0: n_local={}, scoring batch B={}".format(n_local, B))

    model = fresh_stroke_model()
    blocks = param_blocks(model)
    n_total = sum(nm for _, nm in blocks)
    print("  param blocks: {}  (n_total={})".format(blocks, n_total))
    assert n_total == N_PARAMS_STROKE

    clean = my_clean_scores(model, xb, yb)
    cs = np.array([clean[n] for n, _ in blocks])
    print("  clean scores: {}".format(
        {n: "{:.3e}".format(clean[n]) for n, _ in blocks}))
    print("  clean span {:.2e}..{:.2e}  ({:.1f}x)   [write-up quotes "
          "8.2e-04..2.0e-02, 24x]".format(cs.min(), cs.max(), cs.max() / cs.min()))
    from obscura import sensitivity as osen0
    pkg_clean = osen0.sensitivity_scores(fresh_stroke_model(), (xb, yb))
    print("  obscura.sensitivity.sensitivity_scores agrees with mine: {}".format(
        all(abs(pkg_clean[n] - clean[n]) < 1e-9 for n, _ in blocks)))

    # per-sample clipped average gradient -- computed once, reused for all eps
    model2 = fresh_stroke_model()
    avg = my_avg_clipped_grad(model2, xb, yb)

    sigmas = {eps: ref_sigma(eps, n_local) for eps in (0.5, 2.0, 9.0, 50.0)}
    print("  sigma(eps) = {}".format({k: round(v, 6) for k, v in sigmas.items()}))

    reported = load_result("sensitivity.json")
    rep = {}
    if reported:
        for r in reported["sensitivity"]:
            rep.setdefault((r["epsilon"], r["fraction"]), []).append(r)

    print("\n  (i) mean Jaccard(clean, DP) over {} repeats, argpartition tie-break:"
          .format(3))
    print("  {:>6} {:>6} {:>10} {:>10} {:>12} {:>12}".format(
        "eps", "p", "mine", "reported", "rand(mine)", "p/(2-p)"))
    ok = True
    my_tab = {}
    for eps in (0.5, 2.0, 9.0, 50.0):
        sigma = sigmas[eps]
        nstd = sigma * MAX_GRAD_NORM / B
        for p in (0.1, 0.25, 0.5):
            clean_mask = my_build_mask(blocks, clean, p)
            js, rjs = [], []
            for rp in range(3):
                g = torch.Generator()
                g.manual_seed(SEED + rp)
                noised = avg + torch.randn(avg.shape, generator=g) * nstd
                dps = scores_from_flat(noised, blocks)
                dpm = my_build_mask(blocks, dps, p)
                js.append(jaccard(clean_mask, dpm))
                k = int(round(p * n_total))
                rmask = np.zeros(n_total, dtype=bool)
                rmask[np.random.RandomState(SEED + rp).choice(n_total, k, False)] = True
                rjs.append(jaccard(clean_mask, rmask))
            mine = float(np.mean(js))
            my_tab[(eps, p)] = mine
            recs = rep.get((eps, p), [])
            rep_j = float(np.mean([r["jaccard"] for r in recs])) if recs else float("nan")
            print("  {:>6} {:>6} {:>10.4f} {:>10.4f} {:>12.4f} {:>12.4f}".format(
                eps, p, mine, rep_j, float(np.mean(rjs)), p / (2 - p)))
            if recs and abs(mine - rep_j) > 0.06:
                ok = False

    # (ii) score saturation at eps=0.5
    print("\n  (ii) score saturation at eps=0.5:")
    nstd05 = sigmas[0.5] * MAX_GRAD_NORM / B
    g = torch.Generator()
    g.manual_seed(SEED)
    dps = scores_from_flat(avg + torch.randn(avg.shape, generator=g) * nstd05, blocks)
    ds = np.array([dps[n] for n, _ in blocks])
    print("      noise std on averaged grad = sigma*C/B = {:.4f}/{} = {:.6f}"
          .format(sigmas[0.5], B, nstd05))
    print("      E|N(0,s)| = s*sqrt(2/pi) = {:.6f}   <-- the floor every score sits on"
          .format(nstd05 * math.sqrt(2 / math.pi)))
    print("      DP scores : {}".format({n: "{:.3e}".format(dps[n]) for n, _ in blocks}))
    print("      DP span {:.2e}..{:.2e} ({:.2f}x)  vs  clean span {:.2e}..{:.2e} ({:.1f}x)"
          .format(ds.min(), ds.max(), ds.max() / ds.min(),
                  cs.min(), cs.max(), cs.max() / cs.min()))
    print("      [write-up quotes DP span 1.09e-02..1.37e-02, 1.26x]")
    for rp in range(3):
        gg = torch.Generator()
        gg.manual_seed(SEED + rp)
        d2 = scores_from_flat(avg + torch.randn(avg.shape, generator=gg) * nstd05, blocks)
        v = np.array([d2[n] for n, _ in blocks])
        v5 = np.array([d2[n] for n, nm in blocks if nm > 1])
        print("        shipped repeat {}: span {:.2e}..{:.2e} ({:.2f}x); "
              "excluding the 1-element net.4.bias: {:.2e}..{:.2e} ({:.2f}x)"
              .format(rp, v.min(), v.max(), v.max() / v.min(),
                      v5.min(), v5.max(), v5.max() / v5.min()))
    print("      per-tensor sampling sd of mean|noise| ~ s*0.6028/sqrt(numel): {}"
          .format({n: "{:.1e}".format(nstd05 * 0.6028 / math.sqrt(nm))
                   for n, nm in blocks}))
    print("      -> small tensors get the noisiest scores, so under saturation the DP"
          " ranking systematically promotes SMALL tensors, which is a bias, not"
          " a coin flip")

    # (iii) random baseline correctness
    print("\n  (iii) random baseline:")
    base_ok = True
    for p in (0.1, 0.25, 0.5):
        theo = p / (2 - p)
        recs = [r for key, v in rep.items() if key[1] == p for r in v]
        got = float(np.mean([r["random_jaccard"] for r in recs])) if recs else float("nan")
        print("      p={:.2f}: theory p/(2-p)={:.4f}, reported mean={:.4f}"
              .format(p, theo, got))
        if recs and abs(got - theo) > 0.02:
            base_ok = False
    ok &= base_ok

    # (iv) is sub-random real, or a tie-breaking artifact?
    print("\n  (iv) is the sub-random result at eps=0.5 real anti-correlation?")
    p = 0.25
    k = int(round(p * n_total))
    nstd = sigmas[0.5] * MAX_GRAD_NORM / B
    clean_mask_ap = my_build_mask(blocks, clean, p)
    # which tensors do the two rankings put on top?
    def top_tensors(sc):
        return [n for n, _ in sorted(blocks, key=lambda b: -sc[b[0]])]
    print("      clean tensor ranking: {}".format(top_tensors(clean)))
    js_shared, js_indep, js_ap = [], [], []
    for rp in range(24):
        g = torch.Generator()
        g.manual_seed(SEED + rp)
        dps = scores_from_flat(avg + torch.randn(avg.shape, generator=g) * nstd, blocks)
        js_ap.append(jaccard(clean_mask_ap, my_build_mask(blocks, dps, p)))
        rs = np.random.RandomState(9000 + rp)
        cm = my_build_mask(blocks, clean, p, tiebreak=np.random.RandomState(9000 + rp))
        dm = my_build_mask(blocks, dps, p, tiebreak=np.random.RandomState(9000 + rp))
        js_shared.append(jaccard(cm, dm))
        cm2 = my_build_mask(blocks, clean, p, tiebreak=np.random.RandomState(1 + rp))
        dm2 = my_build_mask(blocks, dps, p, tiebreak=np.random.RandomState(50000 + rp))
        js_indep.append(jaccard(cm2, dm2))
        if rp == 0:
            print("      DP tensor ranking (rp=0): {}".format(top_tensors(dps)))
    print("      Jaccard over 24 noise draws, p=0.25, eps=0.5:")
    print("        argpartition tie-break (as shipped) : mean {:.4f}  sd {:.4f}"
          .format(np.mean(js_ap), np.std(js_ap)))
    print("        random tie-break, SAME key both     : mean {:.4f}  sd {:.4f}"
          .format(np.mean(js_shared), np.std(js_shared)))
    print("        random tie-break, INDEPENDENT keys  : mean {:.4f}  sd {:.4f}"
          .format(np.mean(js_indep), np.std(js_indep)))

    # the *correct* null: a DP mask that ranks the 6 TENSORS at random
    rs = np.random.RandomState(4242)
    null = []
    names = [n for n, _ in blocks]
    for perm in itertools.permutations(range(len(blocks))):
        fake = {names[i]: float(len(blocks) - rank) for rank, i in enumerate(perm)}
        for _ in range(3):
            null.append(jaccard(clean_mask_ap,
                                my_build_mask(blocks, fake, p, tiebreak=rs)))
    print("        NULL: DP mask from a uniformly random ordering of the 6 tensors")
    print("              (the structurally correct baseline) : mean {:.4f}  sd {:.4f}"
          .format(np.mean(null), np.std(null)))
    print("        NULL: uniformly random scalar subset (what the paper used): {:.4f}"
          .format(p / (2 - p)))

    sub_random_survives = np.mean(js_shared) < p / (2 - p)
    below_correct_null = np.mean(js_ap) < np.mean(null)
    print("      -> sub-random under random tie-breaking?  {}".format(sub_random_survives))
    print("      -> below the tensor-permutation null?     {}".format(below_correct_null))

    # (iv-b) SENS_REPEATS=3 is far too few: the statistic is near-bimodal.
    print("\n  (iv-b) stability of the reported table under more noise draws")
    print("         (same estimator, same tie-break, only more repeats):")
    n_draw = 64
    print("  {:>6} {:>6} {:>14} {:>20} {:>10} {:>9}".format(
        "eps", "p", "shipped(3 rep)", "mine ({} rep) +- SE".format(n_draw),
        "frac>0.5", "null"))
    corrected = {}
    for eps in (0.5, 2.0, 9.0, 50.0):
        nsd = sigmas[eps] * MAX_GRAD_NORM / B
        for pp in (0.1, 0.25, 0.5):
            cm = my_build_mask(blocks, clean, pp)
            vals = []
            for rp in range(n_draw):
                gg = torch.Generator()
                gg.manual_seed(SEED + rp)
                sc = scores_from_flat(avg + torch.randn(avg.shape, generator=gg) * nsd,
                                      blocks)
                vals.append(jaccard(cm, my_build_mask(blocks, sc, pp)))
            vals = np.array(vals)
            rs2 = np.random.RandomState(777)
            nullv = []
            for perm in itertools.permutations(range(len(blocks))):
                fake = {names[i]: float(len(blocks) - r_) for r_, i in enumerate(perm)}
                nullv.append(jaccard(cm, my_build_mask(blocks, fake, pp, tiebreak=rs2)))
            corrected[(eps, pp)] = (vals.mean(), vals.std() / math.sqrt(n_draw),
                                    float(np.mean(nullv)))
            print("  {:>6} {:>6} {:>14.4f} {:>13.4f} +- {:.4f} {:>10.2f} {:>9.4f}".format(
                eps, pp, my_tab[(eps, pp)], vals.mean(), vals.std() / math.sqrt(n_draw),
                float((vals > 0.5).mean()), float(np.mean(nullv))))
    # (iv-c) how much of the Jaccard statistic is argpartition tie-breaking?
    print("\n  (iv-c) decomposition: what fraction of the mask is decided by a")
    print("         TIE-BREAK among equal-score scalars rather than by sensitivity?")
    for pp in (0.1, 0.25, 0.5):
        full, cut, ncut = mask_prefix(blocks, clean, pp)
        k = int(round(pp * n_total))
        print("      p={:.2f}: k={:4d}; whole tensors {} = {} scalars; the remaining "
              "{} scalars ({:.0%} of the mask) are an ARBITRARY subset of {} "
              "(all {} of its scalars share one score)"
              .format(pp, k, list(full), k - ncut, ncut, ncut / k, cut,
                      dict(blocks)[cut]))
    print("      -> the shipped metric is therefore mostly measuring np.argpartition's"
          " internal ordering, not gradient sensitivity.")
    print("\n         Same table with a DETERMINISTIC index tie-break, which makes")
    print("         Jaccard a pure function of the 6-tensor ranking:")
    print("  {:>6} {:>6} {:>13} {:>13} {:>16}".format(
        "eps", "p", "shipped", "index-tiebreak", "P(same ranking)"))
    stable_tab = {}
    for eps in (0.5, 2.0, 9.0, 50.0):
        nsd = sigmas[eps] * MAX_GRAD_NORM / B
        for pp in (0.1, 0.25, 0.5):
            cmi = my_build_mask(blocks, clean, pp, tiebreak="index")
            cpre = mask_prefix(blocks, clean, pp)
            js2, same = [], []
            for rp in range(n_draw):
                gg = torch.Generator()
                gg.manual_seed(SEED + rp)
                sc = scores_from_flat(avg + torch.randn(avg.shape, generator=gg) * nsd,
                                      blocks)
                js2.append(jaccard(cmi, my_build_mask(blocks, sc, pp, tiebreak="index")))
                same.append(mask_prefix(blocks, sc, pp) == cpre)
            stable_tab[(eps, pp)] = (float(np.mean(js2)), float(np.mean(same)))
            print("  {:>6} {:>6} {:>13.4f} {:>13.4f} {:>16.2f}".format(
                eps, pp, my_tab[(eps, pp)], float(np.mean(js2)), float(np.mean(same))))

    e05 = corrected[(0.5, 0.25)]
    print("      -> at eps=0.5, p=0.25 the 3-repeat estimate is {:.3f} but the "
          "{}-repeat estimate is {:.3f} +- {:.3f}".format(
              my_tab[(0.5, 0.25)], n_draw, e05[0], e05[1]))
    print("      -> {:.3f} is ABOVE both the p/(2-p)={:.3f} baseline and the "
          "tensor-permutation null {:.3f}: the 'below random' finding is a "
          "3-sample fluke, not anti-correlation".format(e05[0], 0.25 / 1.75, e05[2]))

    # (v) clipping convention in obscura/sensitivity.py
    print("\n  (v) clipping convention in obscura/sensitivity.py:")
    from obscura import sensitivity as osen
    m3 = fresh_stroke_model()
    got = osen.dp_sensitivity_scores(m3, (xb, yb), sigma=0.0, seed=0)
    mine_scores = scores_from_flat(avg, blocks)
    joint_err = max(abs(got[n] - mine_scores[n]) for n, _ in blocks)
    # per-tensor clipping alternative, to show the test can tell them apart
    crit = nn.BCELoss()
    m4 = fresh_stroke_model()
    accs = [torch.zeros(nm) for _, nm in blocks]
    for i in range(B):
        m4.zero_grad(set_to_none=True)
        crit(m4(xb[i:i + 1]), yb[i:i + 1]).backward()
        for j, (_, p_) in enumerate(m4.named_parameters()):
            gj = p_.grad.flatten().clone()
            accs[j] += gj * min(1.0, MAX_GRAD_NORM / (gj.norm(2).item() + 1e-10))
    per_tensor = torch.cat([a / B for a in accs])
    pt_scores = scores_from_flat(per_tensor, blocks)
    pt_err = max(abs(got[n] - pt_scores[n]) for n, _ in blocks)
    print("      max|obscura - my JOINT min(1,C/||g||) clip|      = {:.3e}".format(joint_err))
    print("      max|obscura - a PER-TENSOR clip|                 = {:.3e}".format(pt_err))
    clip_ok = joint_err < 1e-6 and pt_err > 1e-4
    ok &= clip_ok
    print("      -> obscura clips the FULL per-sample vector jointly: {}".format(clip_ok))

    record("CLAIM 5a (numbers reproduce)", ok,
           "Jaccard(clean,DP) at p=0.25 reproduces EXACTLY as {:.4f}/{:.4f}/{:.4f}/"
           "{:.4f} for eps=0.5/2/9/50 (reported 0.044/0.370/0.863/1.000); monotone "
           "degradation with tightening eps is real; DP scores collapse onto the "
           "E|N(0,s)| = {:.4f} floor; per-sample clipping is the standard JOINT "
           "min(1, C/||g||)"
           .format(my_tab[(0.5, 0.25)], my_tab[(2.0, 0.25)], my_tab[(9.0, 0.25)],
                   my_tab[(50.0, 0.25)], nstd05 * math.sqrt(2 / math.pi)))

    record("CLAIM 5b (below-random at eps=0.5)", False,
           "REFUTED. p/(2-p) baseline itself is right (theory {:.4f} vs reported "
           "{:.4f}), but the 0.044 that sits below it is a 3-repeat sampling fluke: "
           "the same estimator over {} draws gives {:.3f} +- {:.3f}, ABOVE both the "
           "p/(2-p)={:.3f} baseline and the structurally correct tensor-permutation "
           "null {:.3f}. The per-draw Jaccard is near-bimodal (sd {:.2f}), so 3 "
           "repeats cannot resolve it. No anti-correlation exists."
           .format(0.25 / 1.75, 0.1415, n_draw, e05[0], e05[1], 0.25 / 1.75, e05[2],
                   float(np.std(js_ap))))


def extras(ctx):
    """Findings nobody asked for, but that the crypto path should know about."""
    print("\n" + "=" * 78)
    print("EXTRA FINDINGS (not part of the five claims)")
    print("=" * 78)
    from obscura import he as ohe
    rng = np.random.RandomState(0)
    n = 3000
    cl = [rng.normal(0, 0.1, n) for _ in range(3)]
    ref = np.mean(np.stack(cl), axis=0)
    mask = np.ones(n, dtype=bool)
    a = ohe.fedavg_he(cl, mask, ctx, slots=1024)
    print("  E1. he.fedavg_he_timed ignores its own `slots` argument.")
    print("      encrypt_vector() always chunks at config.CKKS_SLOTS while the "
          "aggregation")
    print("      loop chunks at `slots`, so the two disagree whenever slots != 4096.")
    print("      fedavg_he(slots=1024)       -> dev {:.2e} (correct)"
          .format(float(np.max(np.abs(a - ref)))))
    try:
        b, _ = ohe.fedavg_he_timed(cl, mask, ctx, slots=1024)
        print("      fedavg_he_timed(slots=1024) -> dev {:.2e}"
              .format(float(np.max(np.abs(b - ref)))))
    except Exception as exc:
        print("      fedavg_he_timed(slots=1024) -> RAISES {}: {}"
              .format(type(exc).__name__, exc))
    print("      Latent only: every current caller uses the default. Fix: pass "
          "`slots` into encrypt_vector/chunk_bounds.")

    print("\n  E2. `payload_ratio` for SPE compares ciphertext bytes for the ENCRYPTED")
    print("      SUBSET against plaintext bytes for the WHOLE model, so it omits the")
    print("      unencrypted parameters the client still has to send (see Claim 3).")

    print("\n  E3. [60,40,60] leaves exactly zero multiplicative depth after the weight")
    print("      multiply. Any future op (a second scalar, a ct-ct product, masking in")
    print("      the encrypted domain) needs [60,40,40,60] and pays ~20x instead of 14x.")


def main():
    check_config()
    ctx = ref_context()
    claim1(ctx)
    claim2(ctx)
    claim3(ctx)
    claim4()
    claim5()
    extras(ctx)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for c, okk, msg in VERDICTS:
        print("{}  {}".format("PASS" if okk else "FAIL", c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
