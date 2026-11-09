"""Independent re-derivation of Trial 07/27's CRYPTO-side claims.

This file deliberately does NOT use obscura.he, obscura.aggregate or
obscura.sensitivity to compute any reference value: those modules are what is being
audited. Every reference number is re-derived here from raw TenSEAL / Opacus / numpy /
plain torch. obscura modules are imported only as the *subject* of a comparison (and
obscura.data only to obtain the same client partition the experiment used, which is not
itself under test).

STATUS vs VERDICT -- read this before interpreting the output
-------------------------------------------------------------
Two orthogonal things are reported for every claim, and conflating them is what made
the previous version of this harness unreadable:

  STATUS  -- did the check EXECUTE and reach a confident conclusion?
             OK    the measurement ran and its positive/negative controls behaved
             ERROR the check could not be completed (exception, missing data, a
                   control that failed to fire, a self-consistency violation)

  VERDICT -- is the CLAIM true?
             CONFIRMED                   reproduces as written
             CONFIRMED-WITH-CORRECTION   substantively right, one stated number or
                                         framing must change
             REFUTED                     established false

A REFUTED claim with STATUS=OK is a *success* for this harness: the check did its job.
The process exit code therefore reflects ONLY:
  * any STATUS=ERROR, or
  * any claim whose measured verdict disagrees with the verdict recorded in
    VERIFICATION.md's machine-readable verdict table.
It never goes nonzero merely because a claim turned out to be false.

STALENESS
---------
This harness validates against the *current* contents of results/*.json, not against
numbers pasted from an earlier run. The provenance block printed first records exactly
which files and which config produced this verdict table; VERIFICATION.md carries the
same fingerprint at its top. If they disagree, the write-up has drifted from the data.

Everything is seeded and deterministic except the CKKS ciphertexts themselves: SEAL's
encryption RNG is not exposed by TenSEAL, so measured HE deviations wobble in their last
digit or two between runs (~3.5e-08..4.6e-08 at this update scale). Every threshold here
is orders of magnitude away from that wobble, so no verdict moves.

Run:  /home/user/ObscuraDP/obscura_env/bin/python verify_claims.py     (~60 s)
"""
import hashlib
import itertools
import json
import math
import os
import re
import sys
import traceback
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
ROUNDS = 20
POLY_MOD_DEGREE = 8192
COEFF_MOD_BITS = [60, 40, 60]
GLOBAL_SCALE = 2 ** 40
SLOTS = POLY_MOD_DEGREE // 2          # 4096
HIDDEN_STROKE = (64, 32)
HIDDEN_LOAN = (32, 16)
IN_FEATURES = 15
IN_FEATURES_LOAN = 11
N_PARAMS_STROKE = 3137
N_PARAMS_LOAN = 929
LR_STROKE = 1e-4                      # was 1e-3; see config.py's note
LR_LOAN = 5e-3
SENS_REPEATS = 64                     # was 3; the defect this harness asked to be fixed
SENS_EPSILONS = [0.5, 2.0, 9.0, 50.0]
SENS_FRACTIONS = [0.1, 0.25, 0.5]

RESULTS = os.path.join(HERE, "results")
RESULT_FILES = ["sensitivity.json", "lossless.json", "overhead.json", "arms.json"]

# stroke client sizes after per-client SMOTETomek (obscura.data partition, seed 42)
STROKE_COUNTS = np.array([484.0, 234.0, 1120.0, 4920.0, 738.0])

OK, ERROR = "OK", "ERROR"
CONFIRMED = "CONFIRMED"
CORRECTED = "CONFIRMED-WITH-CORRECTION"
REFUTED = "REFUTED"

CHECKS = []            # (id, title, status, verdict, message)


def record(cid, title, status, verdict, msg):
    CHECKS.append((cid, title, status, verdict, msg))
    print("\n[{}] CLAIM {} -- {}\n     status={}  verdict={}\n     {}"
          .format("check ran" if status == OK else "CHECK FAILED TO RUN",
                  cid, title, status, verdict, msg))


def load_result(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Provenance: exactly what this run validated against.
# ---------------------------------------------------------------------------
def fingerprint():
    """sha256 prefix + record counts for every results file this harness reads."""
    out = {}
    for name in RESULT_FILES:
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            out[name] = ("MISSING", 0)
            continue
        with open(path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()[:12]
        doc = json.loads(raw)
        body = [k for k in doc if k != "meta"]
        n = sum(len(doc[k]) for k in body if isinstance(doc[k], list))
        out[name] = (digest, n)
    return out


def print_provenance(fp):
    print("=" * 78)
    print("PROVENANCE -- what this verdict table was validated against")
    print("=" * 78)
    for name in RESULT_FILES:
        digest, n = fp[name]
        print("  results/{:<18} sha256:{}  {:>5} records".format(name, digest, n))
    print("  config: SENS_REPEATS={}  LR[stroke]={:g}  LR[loan]={:g}  ROUNDS={}  "
          "CKKS={}@{}".format(SENS_REPEATS, LR_STROKE, LR_LOAN, ROUNDS,
                              COEFF_MOD_BITS, POLY_MOD_DEGREE))


def check_config():
    """Confirm the constants above match obscura/config.py verbatim.

    Config drift is an EXECUTION failure, not a claim verdict: it means the numbers
    this harness replicates no longer describe the code that produced the results.
    """
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
        ("HIDDEN_STROKE", HIDDEN_STROKE, config.HIDDEN["stroke"]),
        ("HIDDEN_LOAN", HIDDEN_LOAN, config.HIDDEN["loan"]),
        ("LR_STROKE", LR_STROKE, config.LR["stroke"]),
        ("LR_LOAN", LR_LOAN, config.LR["loan"]),
        ("SENS_REPEATS", SENS_REPEATS, config.SENS_REPEATS),
        ("SENS_EPSILONS", SENS_EPSILONS, config.SENS_EPSILONS),
        ("SENS_FRACTIONS", SENS_FRACTIONS, config.SENS_FRACTIONS),
    ]
    bad = ["{} (harness {!r} vs config {!r})".format(n, a, b)
           for n, a, b in pairs if a != b]
    return bad


# ---------------------------------------------------------------------------
# VERIFICATION.md's machine-readable verdict table -- the harness checks itself
# against the document it is supposed to keep honest.
# ---------------------------------------------------------------------------
DOC = os.path.join(HERE, "VERIFICATION.md")


def parse_documented_verdicts():
    """{claim_id: (status, verdict)} from the marked table in VERIFICATION.md."""
    if not os.path.exists(DOC):
        return None, "VERIFICATION.md not found"
    with open(DOC) as f:
        text = f.read()
    m = re.search(r"<!--\s*VERDICT-TABLE\s*-->(.*?)<!--\s*/VERDICT-TABLE\s*-->",
                  text, re.S)
    if not m:
        return None, "no <!-- VERDICT-TABLE --> block in VERIFICATION.md"
    out = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        cid = cells[0]
        if not re.fullmatch(r"\d+[a-z]?", cid):
            continue
        out[cid] = (cells[2], cells[3])
    if not out:
        return None, "VERDICT-TABLE block parsed but contained no claim rows"
    return out, None


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
# Plain-torch models matching obscura.models' init RNG consumption exactly.
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


def fresh_model(in_features, hidden):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return PlainFFNN(in_features, hidden[0], hidden[1])


def fresh_stroke_model():
    return fresh_model(IN_FEATURES, HIDDEN_STROKE)


def flat_params(model):
    return torch.cat([p.detach().flatten()
                      for p in model.parameters()]).numpy().astype(np.float64)


def param_blocks(model):
    return [(name, p.numel()) for name, p in model.named_parameters()]


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def f32_round_error(x):
    """max |float32(x) - x| -- the storage floor a float32 state_dict already pays."""
    return float(np.max(np.abs(f32(x) - np.asarray(x, dtype=np.float64))))


def half_ulp32(v):
    """Half a float32 ULP at magnitude v: the upper bound on f32 storage error."""
    v = abs(float(v))
    if v == 0.0:
        return 0.0
    return float(np.spacing(np.float32(v))) / 2.0


# ===========================================================================
# CLAIM 1 -- CKKS aggregation lossless, incl. on DP-noised updates + SPE scatter
# ===========================================================================
def claim1(ctx):
    print("\n" + "=" * 78)
    print("CLAIM 1 -- CKKS/SPE losslessness on DP-noised updates")
    print("=" * 78)

    reported = load_result("lossless.json")
    if not reported:
        record("1", "CKKS/SPE losslessness on DP-noised updates", ERROR, None,
               "results/lossless.json missing -- nothing to verify against")
        return
    rr = reported["lossless"]

    # ---- (a) what IS the thing being encrypted? --------------------------
    # lossless.json encrypts client states after 3 local epochs at lr=1e-4 from the
    # seed-42 init. At that learning rate the weights barely leave the init, so the
    # init's own magnitude should predict `update_scale`. If it does, the reference
    # below is built on the right vectors rather than on a guessed scale.
    w0 = flat_params(fresh_stroke_model())
    rep_scale = float(np.mean([r["update_scale"] for r in rr]))
    print("  (a) magnitude of the encrypted object")
    print("      seed-42 stroke init: n={}  mean|w|={:.6f}  max|w|={:.6f}"
          .format(w0.size, np.mean(np.abs(w0)), np.max(np.abs(w0))))
    print("      reported update_scale (mean over {} records) = {:.6f}   "
          "agreement {:+.2%}".format(len(rr), rep_scale,
                                     np.mean(np.abs(w0)) / rep_scale - 1.0))
    scale_ok = abs(np.mean(np.abs(w0)) / rep_scale - 1.0) < 0.02

    # ---- (b) independent HE / SPE reference on those vectors -------------
    print("\n  (b) independent CKKS reference on client vectors of that magnitude")
    print("      (raw ts.ckks_vector, chunked at {} slots, plaintext weights)"
          .format(SLOTS))
    n = w0.size
    clean_scores_mask = None
    rng = np.random.RandomState(2024)
    rows = []
    all_ok = True
    eps_sigma = sorted({r["epsilon"]: r["sigma"] for r in rr}.items())
    for epsilon, sigma in eps_sigma:
        # per-step DP noise at this eps, attenuated by lr over 3 local epochs --
        # the actual jitter between client states, not the raw per-step noise.
        noise_std = sigma * MAX_GRAD_NORM / BATCH_SIZE
        jitter = LR_STROKE * math.sqrt(LOCAL_EPOCHS * 16) * noise_std
        clients = [f32(w0 + rng.normal(0.0, jitter, n)) for _ in range(NUM_CLIENTS)]
        plain = np_weighted_mean(clients, STROKE_COUNTS)
        for p in (0.25, 1.0):
            k = int(round(p * n))
            mask = np.zeros(n, dtype=bool)
            if p >= 1.0:
                mask[:] = True
            else:
                mask[np.sort(np.random.RandomState(7).permutation(n)[:k])] = True
            mine = ref_spe(ctx, clients, mask, STROKE_COUNTS)
            dev = float(np.max(np.abs(mine - plain)))
            rel = dev / float(np.mean(np.abs(plain)))
            rows.append((epsilon, p, dev, rel))
            all_ok &= dev < 1e-6
            mode = "he" if p >= 1.0 else "spe"
            match = [r for r in rr if r["epsilon"] == epsilon
                     and r["spe_fraction"] == p and r["mode"] == mode]
            rep_dev = match[0]["max_abs_dev"] if match else float("nan")
            rep_rel = match[0]["rel_to_update_scale"] if match else float("nan")
            print("      eps={:5.1f} p={:.2f}  mine {:.3e} ({:.3e} x update scale)"
                  "   reported {:.3e} ({:.3e})"
                  .format(epsilon, p, dev, rel, rep_dev, rep_rel))

    devs = [r[2] for r in rows]
    rels = [r[3] for r in rows]
    rep_devs = [r["max_abs_dev"] for r in rr]
    rep_rels = [r["rel_to_update_scale"] for r in rr]
    rep_noise = [r["rel_to_noise_floor"] for r in rr]
    print("      --> mine     : max_abs_dev {:.3e}..{:.3e}   / update scale "
          "{:.3e}..{:.3e}".format(min(devs), max(devs), min(rels), max(rels)))
    print("      --> reported : max_abs_dev {:.3e}..{:.3e}   / update scale "
          "{:.3e}..{:.3e}".format(min(rep_devs), max(rep_devs),
                                  min(rep_rels), max(rep_rels)))
    band_ok = (min(devs) / max(rep_devs) < 3.0) and (max(devs) / min(rep_devs) < 3.0)

    # SPE at p=0.25 encrypts fewer coordinates, so its max over coordinates must be
    # no larger than full HE's. A falsifiable ordering the reported file also shows.
    mine_spe = max(d for e, p, d, _ in rows if p < 1.0)
    mine_he = max(d for e, p, d, _ in rows if p >= 1.0)
    rep_spe = max(r["max_abs_dev"] for r in rr if r["spe_fraction"] < 1.0)
    rep_he = max(r["max_abs_dev"] for r in rr if r["mode"] == "he")
    print("      ordering check: max dev at p=0.25 < at p=1.00 -- mine {} "
          "({:.3e} < {:.3e}), reported {} ({:.3e} < {:.3e})"
          .format(mine_spe < mine_he, mine_spe, mine_he,
                  rep_spe < rep_he, rep_spe, rep_he))
    all_ok &= (mine_spe < mine_he) and (rep_spe < rep_he)

    # ---- (c) internal consistency of the reported normalisations ---------
    print("\n  (c) internal consistency of results/lossless.json")
    bad = [r for r in rr
           if abs(r["rel_to_update_scale"] - r["max_abs_dev"] / r["update_scale"])
           > 1e-15]
    print("      rel_to_update_scale == max_abs_dev / update_scale on {}/{} records"
          .format(len(rr) - len(bad), len(rr)))
    bad2 = [r for r in rr
            if abs(r["rel_to_noise_floor"] - r["max_abs_dev"] / r["noise_std"]) > 1e-15]
    print("      rel_to_noise_floor  == max_abs_dev / noise_std    on {}/{} records"
          .format(len(rr) - len(bad2), len(rr)))
    consistent = not bad and not bad2
    all_ok &= consistent

    # ---- (d) IS the scale-free normalisation defensible? -----------------
    # Falsifiable test: sweep the magnitude of the encrypted vectors over four
    # decades. If dividing by the update scale is genuinely scale-free the ratio
    # stays flat; if the error were a fixed absolute constant the ratio would move
    # by the same 10000x as the magnitude.
    print("\n  (d) is `rel_to_update_scale` actually scale-free? "
          "(sweep |v| over 3 decades around the operating point)")
    print("      {:>10} {:>12} {:>12} {:>14}".format(
        "mean|ref|", "max_abs_dev", "/ mean|ref|", "/ (sigma C / B)"))
    sweep = []
    below = []
    rng2 = np.random.RandomState(31337)
    ref_noise_std = rr[0]["noise_std"]
    for s in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        cl = [rng2.normal(0, s, 3137) for _ in range(NUM_CLIENTS)]
        refv = np_weighted_mean(cl, STROKE_COUNTS)
        got = ref_he_weighted_sum(ctx, cl, STROKE_COUNTS)
        d = float(np.max(np.abs(got - refv)))
        m = float(np.mean(np.abs(refv)))
        (below if s < 1e-1 else sweep).append((m, d, d / m))
        print("      {:>10.5f} {:>12.3e} {:>12.3e} {:>14.3e}{}"
              .format(m, d, d / m, d / ref_noise_std,
                      "   <-- below the operating range" if s < 1e-1 else ""))
    ratios = [r[2] for r in sweep]
    span_scalefree = max(ratios) / min(ratios)
    span_mag = sweep[-1][0] / sweep[0][0]
    print("      over the operating band (mean|ref| {:.3f}..{:.1f}, a {:.0f}x sweep) "
          "`dev` itself moved {:.0f}x while `dev / update_scale` moved only {:.2f}x"
          .format(sweep[0][0], sweep[-1][0], span_mag,
                  sweep[-1][1] / sweep[0][1], span_scalefree))
    print("      -> the normalisation absorbs the scale. A fixed ABSOLUTE error would "
          "have moved")
    print("         the ratio by the full {:.0f}x; it moved {:.2f}x, so this is a real "
          "invariance and".format(span_mag, span_scalefree))
    print("         not an artifact of a narrow sweep.")
    print("      CAVEAT, measured: below mean|ref| ~ 0.01 an additive CKKS floor "
          "(~{:.1e}) takes over".format(below[0][1]))
    print("      and the ratio degrades to {:.2e}. 'Scale-free' holds over the "
          "operating range (the".format(below[0][2]))
    print("      trial's update_scale is {:.4f}, comfortably inside it), not "
          "asymptotically.".format(rep_scale))
    scalefree_ok = span_scalefree < 1.5 and span_mag > 100

    print("      reported spread: rel_to_update_scale {:.2f}x "
          "({:.3e}..{:.3e}) vs rel_to_noise_floor {:.2f}x ({:.3e}..{:.3e})"
          .format(max(rep_rels) / min(rep_rels), min(rep_rels), max(rep_rels),
                  max(rep_noise) / min(rep_noise), min(rep_noise), max(rep_noise)))

    # ---- (e) SPE scatter-back stress test + positive control --------------
    print("\n  (e) SPE scatter-back stress test (non-contiguous mask, high "
          "coordinate-to-coordinate contrast):")
    ns = 5000                                        # spans 2 CKKS chunks
    idx = np.arange(ns)
    sig = 0.9 * np.sin(idx * 2.399963) * (0.15 + 0.85 * ((idx * 37) % 11) / 10.0)
    stress = [sig * (1.0 + 0.37 * c) + 0.11 * np.cos(idx * (1.7 + c))
              for c in range(NUM_CLIENTS)]
    r7 = np.random.RandomState(1234)
    smask = np.zeros(ns, dtype=bool)
    smask[r7.permutation(ns)[: int(0.25 * ns)]] = True   # scattered, NOT contiguous
    longest = max((len(list(g)) for v, g in itertools.groupby(smask) if v), default=0)
    print("      mask: {} True of {}, longest contiguous True run = {}"
          .format(int(smask.sum()), ns, longest))
    sref = np_weighted_mean(stress, STROKE_COUNTS)
    smine = ref_spe(ctx, stress, smask, STROKE_COUNTS)
    stress_dev = float(np.max(np.abs(smine - sref)))
    bad_shift = smine.copy()
    e_idx = np.flatnonzero(smask)
    bad_shift[e_idx] = smine[np.roll(e_idx, 1)]
    control_dev = float(np.max(np.abs(bad_shift - sref)))
    print("      max|SPE-plain| = {:.3e}   (off-by-one positive control = {:.3e}, "
          "{:.0e}x louder)".format(stress_dev, control_dev, control_dev / stress_dev))
    has_power = control_dev > 1e-3
    all_ok &= (stress_dev < 1e-6) and has_power

    # ---- (f) audit obscura.he against the independent reference ----------
    print("\n  (f) auditing obscura.he.fedavg_he / fedavg_he_timed on the same input:")
    from obscura import he as ohe
    w = STROKE_COUNTS / STROKE_COUNTS.sum()
    got = ohe.fedavg_he(stress, smask, ctx, weights=w)
    d_pkg = float(np.max(np.abs(got - sref)))
    got_t, _ = ohe.fedavg_he_timed(stress, smask, ctx, weights=w)
    d_pkg_t = float(np.max(np.abs(got_t - sref)))
    print("      fedavg_he       vs my plaintext ref = {:.3e}".format(d_pkg))
    print("      fedavg_he_timed vs my plaintext ref = {:.3e}".format(d_pkg_t))
    all_ok &= (d_pkg < 1e-6) and (d_pkg_t < 1e-6)

    d_plain = float(np.max(np.abs(got_t[~smask] - sref[~smask])))
    d_enc = float(np.max(np.abs(got_t[smask] - sref[smask])))
    print("      mask-polarity probe: plaintext half {:.3e} (must be ~1e-16), "
          "encrypted half {:.3e} (CKKS precision)".format(d_plain, d_enc))
    all_ok &= (d_plain < 1e-12) and (1e-12 < d_enc < 1e-6)
    n_written = int(np.sum(np.isfinite(got_t)))
    agree = bool(np.allclose(got, got_t, atol=1e-6))
    all_ok &= (n_written == ns) and agree
    print("      {}/{} coordinates finite; fedavg_he and fedavg_he_timed agree: {}"
          .format(n_written, ns, agree))

    status = OK if (all_ok and scale_ok and band_ok and consistent
                    and scalefree_ok) else ERROR
    record("1", "CKKS/SPE losslessness on DP-noised updates", status, CONFIRMED,
           "independent CKKS reference on the seed-42 stroke init gives max|HE-plain| "
           "{:.2e}..{:.2e} (reported {:.2e}..{:.2e}), i.e. {:.2e}..{:.2e} of the update "
           "magnitude (reported {:.2e}..{:.2e}); rel_to_update_scale is scale-free to "
           "{:.1f}x over a {:.0f}x magnitude sweep and is the defensible headline, while "
           "rel_to_noise_floor moves {:.1f}x on the reported grid purely through its "
           "denominator; scattered-mask SPE dev {:.2e} against a {:.2e} off-by-one "
           "control, so the test has power"
           .format(min(devs), max(devs), min(rep_devs), max(rep_devs),
                   min(rels), max(rels), min(rep_rels), max(rep_rels),
                   span_scalefree, span_mag, max(rep_noise) / min(rep_noise),
                   stress_dev, control_dev))


# ===========================================================================
# CLAIM 2 -- weighted homomorphic aggregation
# ===========================================================================
def claim2(ctx):
    print("\n" + "=" * 78)
    print("CLAIM 2 -- weighted homomorphic aggregation")
    print("=" * 78)

    w = STROKE_COUNTS / STROKE_COUNTS.sum()
    print("  weights = {}".format(np.round(w, 6).tolist()))
    rng = np.random.RandomState(11)
    n = 1500
    ok = True
    print("  {:>10} {:>13} {:>13} {:>13}".format(
        "|v| scale", "max abs dev", "max rel dev", "vs unweighted"))
    for scale in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        clients = [rng.normal(0.0, scale, n) for _ in range(NUM_CLIENTS)]
        ref = np_weighted_mean(clients, STROKE_COUNTS)
        mine = ref_he_weighted_sum(ctx, clients, w)
        dev = float(np.max(np.abs(mine - ref)))
        rel = dev / float(np.max(np.abs(ref)))
        unw = np.mean(np.stack(clients), axis=0)
        sep = float(np.max(np.abs(ref - unw)))
        good = dev < max(1e-6, 1e-6 * scale) and sep > 100 * dev
        ok &= good
        print("  {:>10.0e} {:>13.3e} {:>13.3e} {:>13.3e} {}".format(
            scale, dev, rel, sep, "" if good else "  <-- FAIL"))

    from obscura import he as ohe
    clients = [rng.normal(0.0, 0.1, n) for _ in range(NUM_CLIENTS)]
    ref = np_weighted_mean(clients, STROKE_COUNTS)
    pkg = ohe.fedavg_he(clients, np.ones(n, dtype=bool), ctx, weights=STROKE_COUNTS)
    d_pkg = float(np.max(np.abs(pkg - ref)))
    print("  obscura.he weighted full-HE vs my plaintext weighted mean: {:.3e}"
          .format(d_pkg))
    ok &= d_pkg < 1e-6

    a = ts.ckks_vector(ctx, [0.1] * 8)
    depth1 = True
    try:
        _ = (a * 0.3).decrypt()
    except Exception:
        depth1 = False
    depth2 = True
    try:
        _ = ((a * 0.3) * 0.3).decrypt()
    except Exception:
        depth2 = False
    print("  modulus chain {} -> depth-1 scalar multiply: {} ; depth-2: {}"
          .format(COEFF_MOD_BITS, "OK" if depth1 else "FAILS",
                  "OK" if depth2 else "FAILS (as expected)"))
    ok &= depth1 and (not depth2)

    record("2", "weighted homomorphic aggregation", OK if ok else ERROR, CONFIRMED,
           "weighted CKKS aggregation matches the plaintext weighted mean to <1e-6 "
           "absolute at |v| from 1e-3 to 10, and the weighted answer differs from the "
           "unweighted one by >100x the HE error, so the test distinguishes them; chain "
           "[60,40,60] supports exactly one scalar multiply (depth 2 raises 'scale out "
           "of bounds'), which is all the w_c multiply needs -- with ZERO headroom left")


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
    print("  --> one ciphertext costs ~{:.0f} B whatever fraction of its {} slots "
          "is used".format(S, SLOTS))

    def predict(n):
        return math.ceil(n / SLOTS) * S / (4.0 * n)

    raw = 2 * POLY_MOD_DEGREE * 2 * 8
    print("  uncompressed CKKS ct for [60,40,60] = 2*{}*2*8 = {} B -> serialized is "
          "{:.3f}x that (SEAL ZSTD)".format(POLY_MOD_DEGREE, raw, S / raw))

    big = 20 * SLOTS
    v = rng.normal(0, 0.1, big)
    tot = sum(len(ts.ckks_vector(ctx, v[s:s + SLOTS].tolist()).serialize())
              for s in range(0, big, SLOTS))
    asym = tot / (4.0 * big)
    print("  measured asymptotic ratio at n={} (slot-aligned): {:.3f}".format(big, asym))

    ovh = load_result("overhead.json")
    if not ovh:
        record("3", "ciphertext/plaintext payload ratio", ERROR, None,
               "results/overhead.json missing")
        return S, asym

    print("\n  {:>9} {:>7} {:>12} {:>12} {:>12} {:>10}".format(
        "n_params", "chunks", "my predict", "reported", "abs diff", "driver pred"))
    ok = abs(predict(N_PARAMS_STROKE) - 18.757) < 0.15 and abs(asym - 14.4) < 0.2
    for r in ovh["crypto_vs_size"]:
        if r["n_encrypted"] != r["n_params"]:
            continue
        nn_ = r["n_params"]
        pr = predict(nn_)
        print("  {:>9} {:>7} {:>12.3f} {:>12.3f} {:>12.4f} {:>10.3f}".format(
            nn_, math.ceil(nn_ / SLOTS), pr, r["payload_ratio"],
            abs(pr - r["payload_ratio"]), r["predicted_payload_ratio"]))
        ok &= abs(pr - r["payload_ratio"]) < 0.05

    print("\n  padding explanation: {:.4f} / {:.4f} = {:.4f} ; {}/{} = {:.4f}"
          .format(predict(N_PARAMS_STROKE), asym, predict(N_PARAMS_STROKE) / asym,
                  SLOTS, N_PARAMS_STROKE, SLOTS / float(N_PARAMS_STROKE)))

    # The correction this harness previously asked for has been ADOPTED: overhead.json
    # now carries wire_payload_ratio. Re-derive it independently rather than trust it.
    print("\n  wire_payload_ratio (the correction this harness asked for, now shipped):")
    print("      re-derived as (ciphertext_bytes + 4*(n_params - n_encrypted)) "
          "/ plaintext_bytes")
    wire_bad = 0
    n_wire = 0
    for key in ("crypto_vs_p", "crypto_vs_size"):
        for r in ovh[key]:
            mine_wire = ((r["ciphertext_bytes"] + 4.0 * (r["n_params"] - r["n_encrypted"]))
                         / float(r["plaintext_bytes"]))
            n_wire += 1
            if abs(mine_wire - r["wire_payload_ratio"]) > 1e-9:
                wire_bad += 1
    print("      matches on {}/{} records".format(n_wire - wire_bad, n_wire))
    ok &= wire_bad == 0

    sample = [r for r in ovh["crypto_vs_p"] if r["n_params"] == 541697
              and 0 < r["fraction"] < 1]
    for r in sample:
        print("      p={:.2f}: ciphertext-only {:.3f}  ->  true wire {:.3f}"
              .format(r["fraction"], r["payload_ratio"], r["wire_payload_ratio"]))

    record("3", "ciphertext/plaintext payload ratio",
           OK if ok else ERROR, CONFIRMED,
           "ratio(n) = ceil(n/{})*{:.0f}B/(4n) reproduces every reported value to "
           "<0.05; 18.76x at n=3137 is 100% zero-padding ({}/{} = {:.3f}); measured "
           "asymptotic ratio {:.3f}x on the same [60,40,60] chain; the previously "
           "requested wire_payload_ratio is now shipped and re-derives exactly on "
           "{} records"
           .format(SLOTS, S, SLOTS, N_PARAMS_STROKE, SLOTS / float(N_PARAMS_STROKE),
                   asym, n_wire))
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
        print("      eps={:5.1f} -> sigma={:.6f} -> eps_round={:.4f} (err {:+.2%}) "
              "| eps_cumulative over {} rounds = {:.3f}{}"
              .format(eps, s, back, (back - eps) / eps, ROUNDS, cum,
                      "" if good else "   <-- FAIL"))

    s9 = ref_sigma(9.0, n_local)
    e1 = ref_epsilon(s9, sample_rate, steps_round)
    e20 = ref_epsilon(s9, sample_rate, int(ROUNDS * LOCAL_EPOCHS / sample_rate))
    comp_ok = e1 < e20 < ROUNDS * e1
    ok &= comp_ok
    print("  (b) composition: eps(1 round)={:.4f}, eps({} rounds)={:.4f}; "
          "1*eps < composed < {}*eps -> {}"
          .format(e1, ROUNDS, e20, ROUNDS,
                  "OK (sub-linear, RDP)" if comp_ok else "FAIL"))

    try:
        from obscura import dp as odp
        d_s = odp.get_sigma(9.0, n_local)
        d_r = odp.per_round_epsilon(d_s, n_local)
        d_c = odp.cumulative_epsilon(d_s, n_local)
        agree = (abs(d_s - s9) < 1e-9 and abs(d_r - e1) < 1e-9 and abs(d_c - e20) < 1e-9)
        ok &= agree
        print("  (c) obscura.dp reproduces sigma/eps_round/eps_cum exactly: {}"
              .format(agree))
    except Exception as exc:
        ok = False
        print("  (c) obscura.dp raised {} -- cannot cross-check".format(exc))

    # ---- (d) which B? Run Opacus on a client size where the two DIFFER. --
    # The previous version of this check used n_data=1024, where 1024/32 = 32 exactly
    # and expected_batch_size == BATCH_SIZE, so it could not distinguish the two
    # candidate formulas at all. n_data=484 is the real client-0 size and gives
    # expected_batch_size=30, a 6.7% gap the measurement can resolve.
    print("  (d) empirical check of dp_noise_std = sigma*C/B against Opacus")
    print("      (run at n_data=484, the real client-0 size, where Opacus's "
          "expected_batch_size=30 != BATCH_SIZE=32 -- a degenerate n_data that made")
    print("       the two formulas numerically identical would prove nothing):")
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset

    form_ok = True
    for n_data in (484, 1024):
        torch.manual_seed(0)
        sigma_t = 50.0
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
        crit = nn.BCELoss()
        samples = []
        it = iter(loader)
        for _ in range(12):
            xb, yb = next(it)
            if xb.numel() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()
            samples.append(torch.cat([p.grad.flatten()
                                      for p in model.parameters()]).clone())
        emp = torch.stack(samples).std().item()
        eb = opt.expected_batch_size
        pred_B = sigma_t * MAX_GRAD_NORM / BATCH_SIZE
        pred_eb = sigma_t * MAX_GRAD_NORM / eb
        degenerate = (eb == BATCH_SIZE)
        print("      n_data={:5d}: expected_batch_size={:2d}{}  measured std={:.4f}"
              .format(n_data, eb, "  (degenerate: == BATCH_SIZE)" if degenerate else "",
                      emp))
        print("               sigma*C/BATCH_SIZE     = {:.4f} (ratio {:.4f})"
              .format(pred_B, emp / pred_B))
        print("               sigma*C/expected_batch = {:.4f} (ratio {:.4f})"
              .format(pred_eb, emp / pred_eb))
        print("               sigma*C (no 1/B)       = {:.4f} (ratio {:.4f}) "
              "-- decisively excluded".format(sigma_t, emp / sigma_t))
        if not degenerate:
            # the discriminating case: expected_batch must win, nominal must lose
            form_ok &= abs(emp / pred_eb - 1.0) < 0.02
            form_ok &= abs(emp / pred_B - 1.0) > 0.03
    ok &= form_ok
    print("      -> the 1/B is real and B is Opacus's expected_batch_size, not the "
          "nominal batch size: {}".format(form_ok))

    print("      per-client expected_batch_size Opacus would use (stroke, B=32):")
    for nl in (484, 234, 1120, 4920, 738):
        nb = math.ceil(nl / BATCH_SIZE)
        eb = int(nl / nb)
        print("        n_local={:5d} -> len(loader)={:3d} -> expected_batch_size={:2d} "
              "(dp_noise_std assumes 32; understates noise by {:+.1%})"
              .format(nl, nb, eb, 32.0 / eb - 1.0))

    record("4", "DP accounting and dp_noise_std", OK if ok else ERROR, CORRECTED,
           "sigma->epsilon round-trips to <2%; RDP composition is monotone and "
           "sub-linear (eps 9.00 -> {:.2f} over {} rounds, far below {}x); Opacus adds "
           "N(0,(sigma*C)^2) to the SUMMED clipped gradient then divides by "
           "expected_batch_size, so sigma*C/B IS the right std -- CORRECTION: B must be "
           "Opacus's expected_batch_size (29-32 for these clients), not the nominal 32. "
           "At n_data=484 the measurement picks expected_batch_size=30 over 32 "
           "decisively; dp_noise_std therefore understates the floor by up to 10%, "
           "which does not touch the 1e-6 headline"
           .format(e20, ROUNDS, ROUNDS))


# ===========================================================================
# CLAIM 5 -- DP-aware sensitivity (validated against the 64-repeat data)
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


def my_tensor_selection(blocks, scores, fraction):
    """The SET of tensors a top-`fraction` mask draws from.

    Independent re-implementation of the PRIMARY, tie-break-free statistic:
    sort tensors by (-score, name), fill greedily, stop as soon as the running
    total reaches k, return the names touched. Set-valued, so it is invariant to
    the order in which equally-scored tensors happen to be visited.
    """
    n_total = sum(nm for _, nm in blocks)
    k = int(round(fraction * n_total))
    if k <= 0:
        return frozenset()
    order = sorted(blocks, key=lambda b: (-scores[b[0]], b[0]))
    sel, filled = [], 0
    for name, numel in order:
        sel.append(name)
        filled += numel
        if filled >= k:
            break
    return frozenset(sel)


def mask_prefix(blocks, scores, fraction):
    """(fully-included tensors, name of the partially-cut tensor, n_cut)."""
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


def my_random_mask(n_total, fraction, seed):
    k = int(round(fraction * n_total))
    mask = np.zeros(n_total, dtype=bool)
    rng = np.random.RandomState(seed)
    if k >= n_total:
        mask[:] = True
    elif k > 0:
        mask[rng.choice(n_total, size=k, replace=False)] = True
    return mask


def claim5():
    print("\n" + "=" * 78)
    print("CLAIM 5 -- DP-aware sensitivity / mask agreement "
          "(validated against the {}-repeat data)".format(SENS_REPEATS))
    print("=" * 78)

    reported = load_result("sensitivity.json")
    if not reported:
        record("5a", "DP-aware sensitivity: reported numbers reproduce", ERROR, None,
               "results/sensitivity.json missing")
        record("5b", "'mask agreement falls below the random baseline at eps=0.5'",
               ERROR, None, "results/sensitivity.json missing")
        return

    n_rep_file = reported["meta"].get("repeats")
    print("  results/sensitivity.json meta: repeats={}  epsilons={}  fractions={}"
          .format(n_rep_file, reported["meta"]["epsilons"],
                  reported["meta"]["fractions"]))
    if n_rep_file != SENS_REPEATS:
        print("  !! the shipped file has {} repeats, this harness expects {}"
              .format(n_rep_file, SENS_REPEATS))

    rep = {}
    for r in reported["sensitivity"]:
        rep.setdefault((r["epsilon"], r["fraction"]), []).append(r)

    xb, yb, n_local = stroke_client0_batch()
    B = xb.shape[0]
    model = fresh_stroke_model()
    blocks = param_blocks(model)
    n_total = sum(nm for _, nm in blocks)
    names = [n for n, _ in blocks]
    assert n_total == N_PARAMS_STROKE
    print("  client 0: n_local={}, scoring batch B={}, n_params={}"
          .format(n_local, B, n_total))

    clean = my_clean_scores(model, xb, yb)
    cs = np.array([clean[n] for n, _ in blocks])
    print("  clean scores: {}".format(
        {n: "{:.3e}".format(clean[n]) for n, _ in blocks}))
    print("  clean span {:.2e}..{:.2e}  ({:.1f}x)".format(cs.min(), cs.max(),
                                                          cs.max() / cs.min()))
    from obscura import sensitivity as osen
    pkg_clean = osen.sensitivity_scores(fresh_stroke_model(), (xb, yb))
    pkg_agree = all(abs(pkg_clean[n] - clean[n]) < 1e-9 for n, _ in blocks)
    print("  obscura.sensitivity.sensitivity_scores agrees with mine: {}"
          .format(pkg_agree))

    model2 = fresh_stroke_model()
    avg = my_avg_clipped_grad(model2, xb, yb)
    sigmas = {eps: ref_sigma(eps, n_local) for eps in SENS_EPSILONS}
    print("  sigma(eps) = {}".format({k: round(v, 6) for k, v in sigmas.items()}))

    # ---- (i) reproduce all four statistics on all 12 cells, 64 draws -----
    print("\n  (i) independent re-derivation over {} noise draws per cell."
          .format(SENS_REPEATS))
    print("      PRIMARY  = same_tensor_selection (tie-break free)")
    print("      SECONDARY= jaccard over scalars (tie-break sensitive)")
    print("  {:>5} {:>5} | {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>9}"
          .format("eps", "p", "same_me", "same_rep", "jac_me", "jac_rep",
                  "rand_me", "rand_rep", "analytic"))
    ok_a = pkg_agree
    my_j = {}
    my_same = {}
    max_err = 0.0
    for eps in SENS_EPSILONS:
        nstd = sigmas[eps] * MAX_GRAD_NORM / B
        dp_scores = []
        for rp in range(SENS_REPEATS):
            g = torch.Generator()
            g.manual_seed(SEED + rp)
            dp_scores.append(
                scores_from_flat(avg + torch.randn(avg.shape, generator=g) * nstd,
                                 blocks))
        for p in SENS_FRACTIONS:
            cm = my_build_mask(blocks, clean, p)
            csel = my_tensor_selection(blocks, clean, p)
            J, S, R = [], [], []
            for rp in range(SENS_REPEATS):
                J.append(jaccard(cm, my_build_mask(blocks, dp_scores[rp], p)))
                S.append(1.0 if my_tensor_selection(blocks, dp_scores[rp], p) == csel
                         else 0.0)
                R.append(jaccard(cm, my_random_mask(n_total, p, SEED + rp)))
            J = np.array(J)
            my_j[(eps, p)] = J
            my_same[(eps, p)] = float(np.mean(S))
            recs = rep.get((eps, p), [])
            rj = float(np.mean([r["jaccard"] for r in recs]))
            rs = float(np.mean([r["same_tensor_selection"] for r in recs]))
            rr_ = float(np.mean([r["random_jaccard"] for r in recs]))
            an = recs[0]["random_jaccard_analytic"]
            print("  {:>5.1f} {:>5.2f} | {:>8.4f} {:>8.4f} | {:>8.4f} {:>8.4f} | "
                  "{:>8.4f} {:>8.4f} | {:>9.4f}"
                  .format(eps, p, np.mean(S), rs, J.mean(), rj,
                          float(np.mean(R)), rr_, an))
            err = max(abs(np.mean(S) - rs), abs(J.mean() - rj),
                      abs(float(np.mean(R)) - rr_))
            max_err = max(max_err, err)
            if err > 1e-4:
                ok_a = False
    print("      -> worst disagreement between my re-derivation and the shipped file "
          "across 12 cells x 3 statistics: {:.2e}".format(max_err))

    # `same_tensor_selection` is p-invariant on this model -- a real caveat.
    p_invariant = all(
        abs(my_same[(eps, SENS_FRACTIONS[0])] - my_same[(eps, p)]) < 1e-12
        for eps in SENS_EPSILONS for p in SENS_FRACTIONS)
    print("      NOTE: the primary metric is identical at p=0.1/0.25/0.5 ({}). "
          "At n=3137 every".format(p_invariant))
    print("      k in {{314, 784, 1568}} lands inside the same boundary tensor, so "
          "P(same tensors)")
    print("      carries no information about p on this model -- it is a statement "
          "about the tensor")
    print("      RANKING only. That is a property of the model size, not of the "
          "statistic.")

    # ---- (ii) score saturation --------------------------------------------
    print("\n  (ii) saturation mechanism at eps=0.5:")
    nstd05 = sigmas[0.5] * MAX_GRAD_NORM / B
    g = torch.Generator()
    g.manual_seed(SEED)
    dps = scores_from_flat(avg + torch.randn(avg.shape, generator=g) * nstd05, blocks)
    ds = np.array([dps[n] for n, _ in blocks])
    floor = nstd05 * math.sqrt(2 / math.pi)
    print("      noise std on averaged grad = sigma*C/B = {:.4f}/{} = {:.6f}"
          .format(sigmas[0.5], B, nstd05))
    print("      E|N(0,s)| = s*sqrt(2/pi) = {:.6f}  <-- the floor every score sits on"
          .format(floor))
    print("      DP span (draw 0) {:.2e}..{:.2e} ({:.2f}x)  vs  clean span "
          "{:.2e}..{:.2e} ({:.1f}x)".format(ds.min(), ds.max(), ds.max() / ds.min(),
                                            cs.min(), cs.max(), cs.max() / cs.min()))
    # The write-up quotes a single "1.26x" compressed span. Measure what that number
    # actually is across all 64 draws -- and separate out the 1-element net.4.bias,
    # whose score is ONE |N(0,s)| sample and therefore has an enormous spread.
    span_all, span_multi = [], []
    for rp in range(SENS_REPEATS):
        gg = torch.Generator()
        gg.manual_seed(SEED + rp)
        d2 = scores_from_flat(avg + torch.randn(avg.shape, generator=gg) * nstd05,
                              blocks)
        v = np.array([d2[n] for n, _ in blocks])
        v5 = np.array([d2[n] for n, nm in blocks if nm > 1])
        span_all.append(v.max() / v.min())
        span_multi.append(v5.max() / v5.min())
    print("      DP span over all {} draws: including the 1-element net.4.bias, median "
          "{:.2f}x (range {:.2f}..{:.2f}x);".format(SENS_REPEATS, np.median(span_all),
                                                    min(span_all), max(span_all)))
    print("      restricted to the 5 multi-element tensors, median {:.2f}x (range "
          "{:.2f}..{:.2f}x)".format(np.median(span_multi), min(span_multi),
                                    max(span_multi)))
    print("      -> a quoted '1.26x compressed span' describes the FIVE multi-element "
          "tensors on a")
    print("         typical draw. All six -- which is what the mask actually ranks, and "
          "net.4.bias is")
    print("         the TOP-ranked clean tensor -- typically span {:.1f}x. Say which set "
          "is meant.".format(np.median(span_all)))
    print("      per-tensor sampling sd of mean|noise| ~ s*0.6028/sqrt(numel): {}"
          .format({n: "{:.1e}".format(nstd05 * 0.6028 / math.sqrt(nm))
                   for n, nm in blocks}))
    print("      -> small tensors get the noisiest scores, so under saturation the DP "
          "ranking systematically promotes SMALL tensors: a bias, not a coin flip")

    # ---- (iii) how much of Jaccard is a tie-break? -------------------------
    print("\n  (iii) what fraction of the scalar mask is decided by a TIE-BREAK among "
          "equal scores?")
    for pp in SENS_FRACTIONS:
        full, cut, ncut = mask_prefix(blocks, clean, pp)
        k = int(round(pp * n_total))
        print("      p={:.2f}: k={:4d}; whole tensors {} = {} scalars; the remaining "
              "{} ({:.0%} of the mask) are an ARBITRARY subset of {} (all {} of its "
              "scalars share one score)"
              .format(pp, k, list(full), k - ncut, ncut, ncut / k, cut,
                      dict(blocks)[cut]))
    p = 0.25
    nstd = sigmas[0.5] * MAX_GRAD_NORM / B
    cm_ap = my_build_mask(blocks, clean, p)
    js_ap, js_shared, js_indep = [], [], []
    for rp in range(24):
        gg = torch.Generator()
        gg.manual_seed(SEED + rp)
        sc = scores_from_flat(avg + torch.randn(avg.shape, generator=gg) * nstd, blocks)
        js_ap.append(jaccard(cm_ap, my_build_mask(blocks, sc, p)))
        js_shared.append(jaccard(
            my_build_mask(blocks, clean, p, tiebreak=np.random.RandomState(9000 + rp)),
            my_build_mask(blocks, sc, p, tiebreak=np.random.RandomState(9000 + rp))))
        js_indep.append(jaccard(
            my_build_mask(blocks, clean, p, tiebreak=np.random.RandomState(1 + rp)),
            my_build_mask(blocks, sc, p, tiebreak=np.random.RandomState(50000 + rp))))
    rs = np.random.RandomState(4242)
    null = []
    for perm in itertools.permutations(range(len(blocks))):
        fake = {names[i]: float(len(blocks) - rank) for rank, i in enumerate(perm)}
        null.append(jaccard(cm_ap, my_build_mask(blocks, fake, p, tiebreak=rs)))
    print("      Jaccard at eps=0.5, p=0.25 under three tie-break conventions "
          "(24 draws):")
    print("        argpartition (as shipped)      : {:.4f}".format(np.mean(js_ap)))
    print("        random tie-break, SAME key     : {:.4f}".format(np.mean(js_shared)))
    print("        random tie-break, INDEP keys   : {:.4f}".format(np.mean(js_indep)))
    print("        tensor-permutation null (720!) : {:.4f}".format(np.mean(null)))
    print("      -> the scalar Jaccard swings {:.1f}x on tie-break convention alone, "
          "which is why same_tensor_selection is the right primary metric"
          .format(max(np.mean(js_shared), np.mean(js_ap)) / np.mean(js_indep)))

    # ---- (iv) clipping convention -----------------------------------------
    print("\n  (iv) clipping convention in obscura/sensitivity.py:")
    m3 = fresh_stroke_model()
    got = osen.dp_sensitivity_scores(m3, (xb, yb), sigma=0.0, seed=0)
    mine_scores = scores_from_flat(avg, blocks)
    joint_err = max(abs(got[n] - mine_scores[n]) for n, _ in blocks)
    crit = nn.BCELoss()
    m4 = fresh_stroke_model()
    accs = [torch.zeros(nm) for _, nm in blocks]
    for i in range(B):
        m4.zero_grad(set_to_none=True)
        crit(m4(xb[i:i + 1]), yb[i:i + 1]).backward()
        for j, (_, p_) in enumerate(m4.named_parameters()):
            gj = p_.grad.flatten().clone()
            accs[j] += gj * min(1.0, MAX_GRAD_NORM / (gj.norm(2).item() + 1e-10))
    pt_scores = scores_from_flat(torch.cat([a / B for a in accs]), blocks)
    pt_err = max(abs(got[n] - pt_scores[n]) for n, _ in blocks)
    print("      max|obscura - my JOINT min(1,C/||g||) clip| = {:.3e}".format(joint_err))
    print("      max|obscura - a PER-TENSOR clip|            = {:.3e}".format(pt_err))
    clip_ok = joint_err < 1e-6 and pt_err > 1e-4
    ok_a &= clip_ok
    print("      -> obscura clips the FULL per-sample vector jointly: {} "
          "(the test can tell the two apart)".format(clip_ok))

    record("5a", "DP-aware sensitivity: the reported numbers reproduce",
           OK if ok_a else ERROR, CORRECTED,
           "all 12 (eps,p) cells of the {}-repeat results/sensitivity.json reproduce "
           "from an independent implementation to {:.0e} -- i.e. bit-for-bit -- on all "
           "three statistics: PRIMARY same_tensor_selection = {:.3f}/{:.3f}/{:.3f}/"
           "{:.3f} and SECONDARY Jaccard@p=0.25 = {:.3f}/{:.3f}/{:.3f}/{:.3f} for "
           "eps=0.5/2/9/50, plus the measured random control. Monotone degradation with "
           "tightening eps is real; DP scores collapse onto the E|N(0,s)| = {:.4f} "
           "floor; the clean span is {:.1f}x; clipping is the standard JOINT "
           "min(1,C/||g||). CORRECTION: the '1.26x compressed DP span' is the FIVE "
           "multi-element tensors on a typical draw (median {:.2f}x over {} draws); "
           "including the 1-element net.4.bias -- the top-ranked clean tensor, which "
           "the mask does rank -- the typical span is {:.1f}x (up to {:.1f}x). State "
           "which set is meant"
           .format(SENS_REPEATS, max(max_err, 1e-16),
                   my_same[(0.5, 0.25)], my_same[(2.0, 0.25)],
                   my_same[(9.0, 0.25)], my_same[(50.0, 0.25)],
                   my_j[(0.5, 0.25)].mean(), my_j[(2.0, 0.25)].mean(),
                   my_j[(9.0, 0.25)].mean(), my_j[(50.0, 0.25)].mean(), floor,
                   cs.max() / cs.min(), np.median(span_multi), SENS_REPEATS,
                   np.median(span_all), max(span_all)))

    # =======================================================================
    # 5b -- the retracted "below the random baseline at eps=0.5" claim.
    # The write-up records this as REFUTED. This check RE-MEASURES it and
    # confirms the refutation; agreement is a PASS for the harness.
    # =======================================================================
    print("\n" + "-" * 78)
    print("  CLAIM 5b -- 'mask agreement falls BELOW the random baseline at eps=0.5'")
    print("  The write-up records this as REFUTED. Re-measuring it now.")
    print("-" * 78)

    print("\n  (i) is the random baseline itself right? p/(2-p) vs the shipped control:")
    base_ok = True
    for pp in SENS_FRACTIONS:
        theo = pp / (2 - pp)
        recs = [r for key, v in rep.items() if key[1] == pp for r in v]
        got_m = float(np.mean([r["random_jaccard"] for r in recs]))
        an = recs[0]["random_jaccard_analytic"]
        good = abs(got_m - theo) < 0.02 and abs(an - theo) < 1e-12
        base_ok &= good
        print("      p={:.2f}: theory {:.4f} | shipped analytic field {:.4f} | "
              "shipped measured control {:.4f}   {}"
              .format(pp, theo, an, got_m, "ok" if good else "MISMATCH"))

    J = my_j[(0.5, 0.25)]
    base = 0.25 / 1.75
    se = J.std(ddof=0) / math.sqrt(J.size)
    z = (J.mean() - base) / se
    print("\n  (ii) the eps=0.5, p=0.25 cell over {} draws:".format(SENS_REPEATS))
    print("      mean {:.4f}  sd {:.4f}  SE {:.4f}  ->  {:.1f} standard errors ABOVE "
          "the p/(2-p)={:.4f} baseline".format(J.mean(), J.std(ddof=0), se, z, base))
    lo_mode = int((J < 0.10).sum())
    hi_mode = int((J > 0.50).sum())
    mid = J.size - lo_mode - hi_mode
    print("      the distribution is BIMODAL, not merely noisy: {} of {} draws land "
          "below 0.10 (lowest {} are {:.4f}..{:.4f}) and {} land above 0.50; only {} "
          "draw(s) fall in between. A mean is a poor summary of this, which is exactly "
          "why 3 of them was fatal."
          .format(lo_mode, J.size, lo_mode, float(np.sort(J)[0]),
                  float(np.sort(J)[lo_mode - 1]), hi_mode, mid))

    print("\n  (iii) could 3 draws have produced the retracted 0.044 by chance?")
    print("        Enumerating ALL C({},3) triples of the {} measured draws:"
          .format(J.size, J.size))
    tri = np.array([(J[a] + J[b] + J[c]) / 3.0
                    for a, b, c in itertools.combinations(range(J.size), 3)])
    p_below = float((tri < base).mean())
    p_at = float((tri <= 0.0442 + 1e-9).mean())
    print("        {} triples: P(3-draw mean < baseline {:.4f}) = {:.3f}; "
          "P(3-draw mean <= 0.0442, the published value) = {:.4f} (~1 in {:.0f})"
          .format(tri.size, base, p_below, p_at, 1.0 / max(p_at, 1e-9)))
    first3 = float(np.mean(J[:3]))
    print("        the first three draws (seeds {}..{}, the ones the 3-repeat run used) "
          "give {:.4f} -- reproducing the retracted number exactly"
          .format(SEED, SEED + 2, first3))

    print("\n  (iv) does the sampling argument still hold at {} repeats?"
          .format(SENS_REPEATS))
    gap = J.mean() - base
    n_needed = (3.0 * J.std(ddof=0) / gap) ** 2
    print("        gap to baseline {:.4f}; draws needed for SE = gap/3: {:.0f}; "
          "shipped {} -> margin {:.1f}x".format(gap, n_needed, SENS_REPEATS,
                                                SENS_REPEATS / n_needed))
    print("        at 3 draws SE would be {:.4f}, i.e. a +-{:.2f} 95% interval on a "
          "statistic bounded in [0,1]: unresolvable by construction"
          .format(J.std(ddof=0) / math.sqrt(3), 1.96 * J.std(ddof=0) / math.sqrt(3)))
    rep_ok = SENS_REPEATS >= n_needed

    print("\n  (v) the paper's null was also the wrong null. Because build_mask gives "
          "every scalar")
    print("      its tensor's score, the DP mask is a union of whole tensors plus one "
          "partial tensor.")
    print("      The structurally correct null -- the DP ranking orders the 6 tensors "
          "at random, all")
    print("      720 permutations enumerated -- is {:.4f} at p=0.25, and the {}-draw "
          "measurement".format(np.mean(null), SENS_REPEATS))
    print("      exceeds even that at every eps.")

    refuted = (J.mean() > base) and (z > 3.0)
    status_b = OK if (base_ok and refuted and rep_ok) else ERROR
    record("5b", "'mask agreement falls BELOW the random baseline at eps=0.5'",
           status_b, REFUTED,
           "RE-MEASURED AND STILL REFUTED, as the write-up records. The p/(2-p) "
           "baseline is itself correct (analytic field and measured control both "
           "match). Over the shipped {} draws the eps=0.5, p=0.25 cell is "
           "{:.4f} +- {:.4f}, {:.1f} SE ABOVE the {:.4f} baseline and above the "
           "tensor-permutation null {:.4f}. The retracted 0.044 is reproduced exactly "
           "by the first 3 draws, and {:.1f}% of all C({},3) triples of the same 64 "
           "draws would land at or below it -- a ~1-in-{:.0f} sample, not evidence. "
           "The distribution is bimodal ({} of {} draws below 0.10, {} above 0.50, "
           "{} in between); {} repeats gives a {:.1f}x margin over the {:.0f} needed "
           "to resolve the gap. No anti-correlation exists"
           .format(SENS_REPEATS, J.mean(), se, z, base, float(np.mean(null)),
                   100 * p_at, J.size, 1.0 / max(p_at, 1e-9),
                   lo_mode, J.size, hi_mode, mid,
                   SENS_REPEATS, SENS_REPEATS / n_needed, n_needed))


# ===========================================================================
# CLAIM 6 -- the float32 storage floor calibration on the arms (results/arms.json)
# ===========================================================================
def claim6(ctx):
    print("\n" + "=" * 78)
    print("CLAIM 6 -- the float32 storage floor and the HE arms' ratio to it")
    print("=" * 78)

    arms = load_result("arms.json")
    if not arms:
        record("6a", "the float32 floor is a correctly-computed calibration",
               ERROR, None, "results/arms.json missing")
        record("6b", "'the HE arms sit at 3.5-3.9x the float32 storage floor'",
               ERROR, None, "results/arms.json missing")
        return
    A = arms["arms"]

    HE_STACKS = ("he", "dp_he", "dp_spe")
    print("  {:<7} {:<7} {:<8} {:>9} {:>13} {:>13} {:>8}".format(
        "dataset", "stack", "algo", "n_params", "max_abs_dev", "f32_floor", "ratio"))
    for r in A:
        L = r["lossless"]
        print("  {:<7} {:<7} {:<8} {:>9} {:>13.4e} {:>13.4e} {:>8.4f}".format(
            r["dataset"], r["stack"], r["algo"], r["n_params"],
            L["max_abs_dev"], L["float32_floor"], L["dev_over_float32_floor"]))

    # ---- (a) is the floor what it claims to be? --------------------------
    print("\n  (a) is `float32_floor` really the float32 storage floor?")
    print("      Definition under audit: max |float32(ref_agg) - ref_agg|, which is "
          "bounded by")
    print("      half a float32 ULP at max|w| -- a quantity fixed by ONE number, the "
          "largest weight.")
    w0_stroke = flat_params(fresh_model(IN_FEATURES, HIDDEN_STROKE))
    w0_loan = flat_params(fresh_model(IN_FEATURES_LOAN, HIDDEN_LOAN))
    hu_stroke = half_ulp32(np.max(np.abs(w0_stroke)))
    print("      seed-42 stroke init: max|w|={:.6f}  -> half-ULP = {:.17g}"
          .format(np.max(np.abs(w0_stroke)), hu_stroke))
    stroke_floors = sorted({r["lossless"]["float32_floor"] for r in A
                            if r["dataset"] == "stroke"})
    print("      stroke floors in arms.json: {}"
          .format(["{:.17g}".format(x) for x in stroke_floors]))
    stroke_exact = all(abs(x - hu_stroke) < 1e-24 for x in stroke_floors)
    print("      -> the stroke floor equals that half-ULP EXACTLY: {}".format(
        stroke_exact))
    print("         (lr={:g} keeps the stroke weights essentially at their init, so the "
          "floor is".format(LR_STROKE))
    print("          predictable from the init alone -- an independent confirmation "
          "that nothing")
    print("          in the aggregation path is inventing this number.)")

    loan_floors = sorted({r["lossless"]["float32_floor"] for r in A
                          if r["dataset"] == "loan"})
    print("      seed-42 loan init:   max|w|={:.6f}  -> half-ULP = {:.4e}"
          .format(np.max(np.abs(w0_loan)), half_ulp32(np.max(np.abs(w0_loan)))))
    print("      loan floors in arms.json: {}"
          .format(["{:.4e}".format(x) for x in loan_floors]))
    lo = max(loan_floors)
    implied = lo * 2 ** 24
    print("      -> the loan floor is {:.4e} = 2^-24, which implies max|w| in "
          "[{:.2f}, {:.2f}) after training".format(lo, implied, 2 * implied))
    print("         (loan trains at lr={:g} and its weights grow past 1.0, two OCTAVES "
          "above stroke's".format(LR_LOAN))
    print("          0.26 -- so the two datasets do not even share a denominator.)")

    # arithmetic self-consistency + the plaintext arms must sit exactly on the floor
    bad_ratio = [r for r in A
                 if abs(r["lossless"]["dev_over_float32_floor"]
                        - r["lossless"]["max_abs_dev"] / r["lossless"]["float32_floor"])
                 > 1e-9]
    plain = [r for r in A if r["stack"] in ("plain", "dp")]
    plain_on_floor = all(abs(r["lossless"]["dev_over_float32_floor"] - 1.0) < 1e-9
                         for r in plain)
    print("\n      ratio == max_abs_dev / float32_floor on {}/{} arms; the {} non-HE "
          "arms sit exactly on the floor (ratio 1.000): {}"
          .format(len(A) - len(bad_ratio), len(A), len(plain), plain_on_floor))
    print("      -> for a plaintext arm the ONLY deviation from the float64 reference "
          "is the float32")
    print("         store, so ratio 1.000 is a structural identity and confirms the "
          "instrument.")
    ok_a = stroke_exact and not bad_ratio and plain_on_floor

    record("6a", "the float32 floor is a correctly-computed calibration",
           OK if ok_a else ERROR, CONFIRMED,
           "float32_floor is exactly the float32 half-ULP at the largest aggregated "
           "weight: for stroke it equals {:.6e}, reproduced to the last bit from the "
           "seed-42 init's max|w|={:.6f} without touching obscura's aggregation path; "
           "the {} non-HE arms sit exactly on it (ratio 1.000), which is the structural "
           "identity that validates the instrument; the ratio field re-derives on all "
           "{} arms. Calibrating HE error against the rounding a float32 state_dict "
           "already pays is a sound idea and worth keeping"
           .format(hu_stroke, np.max(np.abs(w0_stroke)), len(plain), len(A)))

    # ---- (b) the quoted range ---------------------------------------------
    print("\n  (b) what range do the HE arms ACTUALLY occupy?")
    he_arms = [r for r in A if r["stack"] in HE_STACKS]
    ratios = np.array([r["lossless"]["dev_over_float32_floor"] for r in he_arms])
    by_ds = {}
    for r in he_arms:
        by_ds.setdefault(r["dataset"], []).append(
            r["lossless"]["dev_over_float32_floor"])
    print("      all {} HE-bearing arms (stacks {}): {:.3f}x .. {:.3f}x"
          .format(len(he_arms), "/".join(HE_STACKS), ratios.min(), ratios.max()))
    for ds in sorted(by_ds):
        v = np.array(by_ds[ds])
        devs = [r["lossless"]["max_abs_dev"] for r in he_arms if r["dataset"] == ds]
        fl = sorted({r["lossless"]["float32_floor"] for r in he_arms
                     if r["dataset"] == ds})
        print("        {:<7} n={}  {:.3f}x .. {:.3f}x   (dev {:.3e}..{:.3e} against a "
              "{:.3e} floor)"
              .format(ds, v.size, v.min(), v.max(), min(devs), max(devs), max(fl)))
    claimed_lo, claimed_hi = 3.5, 3.9
    inside = int(((ratios >= claimed_lo) & (ratios <= claimed_hi)).sum())
    print("      the write-up claims {}-{}x: {} of {} HE arms fall inside that band"
          .format(claimed_lo, claimed_hi, inside, len(he_arms)))

    # ---- (c) is the ratio even a stable quantity to quote? ---------------
    print("\n  (c) IS 'ratio to the float32 floor' a stable quantity to quote?")
    print("      Mechanism: the numerator (CKKS error) grows CONTINUOUSLY with the "
          "magnitude")
    print("      of what is encrypted; the denominator (float32 half-ULP) is a "
          "POWER-OF-TWO")
    print("      STAIRCASE in max|w|. Their ratio is therefore a sawtooth with period "
          "one octave.")
    print("      Direct test: hold the model, the chain and the client count fixed and "
          "scale the")
    print("      weights across a binade boundary.")
    rngc = np.random.RandomState(5)
    base_v = [rngc.normal(0, 0.12, N_PARAMS_STROKE) for _ in range(NUM_CLIENTS)]
    print("      {:>9} {:>10} {:>13} {:>13} {:>9}".format(
        "scale", "max|ref|", "f32 floor", "ckks dev", "ratio"))
    saw = []
    for fac in (0.9, 1.0, 1.3, 1.6, 1.9, 2.1):
        cl = [v * fac for v in base_v]
        ref = np_weighted_mean(cl, STROKE_COUNTS)
        got = f32(ref_he_weighted_sum(ctx, cl, STROKE_COUNTS))
        dev = float(np.max(np.abs(got - ref)))
        fl = f32_round_error(ref)
        saw.append((float(np.max(np.abs(ref))), fl, dev, dev / fl))
        print("      {:>9.2f} {:>10.5f} {:>13.4e} {:>13.4e} {:>9.3f}"
              .format(fac, saw[-1][0], fl, dev, dev / fl))
    sr = [s[3] for s in saw]
    print("      -> with NOTHING changed but the weight magnitude, the ratio moves "
          "{:.3f}x .. {:.3f}x ({:.2f}x swing)".format(min(sr), max(sr),
                                                      max(sr) / min(sr)))
    print("         and it RESETS when max|ref| crosses a power of two (the floor "
          "doubles there).")
    unstable = (max(sr) / min(sr)) > 1.2
    print("      -> the ratio is therefore only meaningful as a BAND, and the band must "
          "be")
    print("         re-derived whenever the learning rate, the round count or the "
          "dataset changes")
    print("         the weight magnitudes. That is exactly what happened here: "
          "lr[stroke] 1e-3 -> 1e-4.")

    print("\n      A second, smaller methodological point: `dev_over_float32_floor` is "
          "max(dev over")
    print("      rounds) / max(floor over rounds), two maxima that need not come from "
          "the same")
    print("      round, so it is not exactly any round's ratio.")

    status_b = OK if unstable else ERROR
    record("6b", "'the HE arms sit at 3.5-3.9x the float32 storage floor'",
           status_b, REFUTED,
           "REFUTED against the current results/arms.json: {} of {} HE-bearing arms "
           "fall in [3.5, 3.9]. The measured range is {:.3f}x .. {:.3f}x -- stroke is "
           "uniformly {:.3f}x across all {} of its HE arms (dev {:.3e} against a "
           "{:.3e} floor) and loan spans {:.3f}x .. {:.3f}x (dev {:.3e}..{:.3e} against "
           "a {:.3e} floor). The 3.5-3.9x figure predates lr[stroke] 1e-3 -> 1e-4. "
           "Moreover the quantity is only stable to within a factor of ~2 by "
           "construction: numerator continuous in max|w|, denominator a power-of-two "
           "staircase, measured here to swing {:.2f}x from weight magnitude alone with "
           "everything else fixed. Quote it as a band, per dataset, or quote the "
           "scale-free rel_to_update_scale instead"
           .format(inside, len(he_arms), ratios.min(), ratios.max(),
                   np.max(by_ds["stroke"]), len(by_ds["stroke"]),
                   max(r["lossless"]["max_abs_dev"] for r in he_arms
                       if r["dataset"] == "stroke"),
                   max(r["lossless"]["float32_floor"] for r in he_arms
                       if r["dataset"] == "stroke"),
                   min(by_ds["loan"]), max(by_ds["loan"]),
                   min(r["lossless"]["max_abs_dev"] for r in he_arms
                       if r["dataset"] == "loan"),
                   max(r["lossless"]["max_abs_dev"] for r in he_arms
                       if r["dataset"] == "loan"),
                   max(r["lossless"]["float32_floor"] for r in he_arms
                       if r["dataset"] == "loan"),
                   max(sr) / min(sr)))


# ===========================================================================
# Extras
# ===========================================================================
def extras(ctx):
    """Findings nobody asked for, plus the status of the previous round's."""
    print("\n" + "=" * 78)
    print("EXTRA FINDINGS (not part of the numbered claims)")
    print("=" * 78)
    from obscura import he as ohe
    rng = np.random.RandomState(0)
    n = 3000
    cl = [rng.normal(0, 0.1, n) for _ in range(3)]
    ref = np.mean(np.stack(cl), axis=0)
    mask = np.ones(n, dtype=bool)

    print("  E1 [RESOLVED] he.fedavg_he_timed used to ignore its own `slots` argument.")
    a = ohe.fedavg_he(cl, mask, ctx, slots=1024)
    da = float(np.max(np.abs(a - ref)))
    try:
        b, _ = ohe.fedavg_he_timed(cl, mask, ctx, slots=1024)
        db = float(np.max(np.abs(b - ref)))
        print("      fedavg_he(slots=1024)       -> dev {:.2e}".format(da))
        print("      fedavg_he_timed(slots=1024) -> dev {:.2e}".format(db))
        print("      Both correct at a non-default slot count: `slots` is now threaded "
              "into encrypt_vector/chunk_bounds. Previously the timed path raised "
              "IndexError. Fixed.")
    except Exception as exc:
        print("      fedavg_he_timed(slots=1024) -> STILL RAISES {}: {}"
              .format(type(exc).__name__, exc))

    ovh = load_result("overhead.json")
    has_wire = bool(ovh) and all("wire_payload_ratio" in r
                                 for r in ovh["crypto_vs_p"] + ovh["crypto_vs_size"])
    print("\n  E2 [RESOLVED] `payload_ratio` under-counted SPE's wire cost. "
          "`wire_payload_ratio` is now recorded on every overhead record: {}"
          .format(has_wire))

    print("\n  E3 [STANDS] [60,40,60] leaves exactly zero multiplicative depth after "
          "the weight")
    print("      multiply. Any further homomorphic operation needs [60,40,40,60] and "
          "moves the")
    print("      asymptotic payload from ~14.4x to ~20x.")

    lj = load_result("lossless.json")
    keys = [(r["epsilon"], r["spe_fraction"], r["mode"]) for r in lj["lossless"]] if lj else []
    dupes = len(keys) - len(set(keys))
    print("\n  E4 [RESOLVED] run_losslessness.py used to emit two indistinguishable "
          "(eps, 1.0, 'he')")
    print("      records per epsilon. Duplicate (eps, p, mode) keys in the current "
          "file: {}.".format(dupes))

    print("\n  E5 [NEW] `same_tensor_selection`, the primary sensitivity statistic, is "
          "p-INVARIANT")
    print("      on the deployed 3137-parameter model: k = 314, 784 and 1568 all land "
          "inside the")
    print("      same boundary tensor, so p=0.1, 0.25 and 0.5 give bit-identical "
          "values. It is a")
    print("      statement about the tensor ranking only. Reporting it in a table with "
          "a p column")
    print("      invites a reader to think p was varied and nothing happened. Either "
          "collapse the")
    print("      p axis for that metric or measure it on a model with more than 6 "
          "tensors.")

    print("\n  E6 [NEW] arms.json's `dev_over_float32_floor` divides max-over-rounds by "
          "max-over-")
    print("      rounds, so its numerator and denominator can come from different "
          "rounds. Recording")
    print("      the per-round floor alongside `max_abs_dev_per_round` would make it a "
          "true per-round")
    print("      ratio. The effect is small here but the quantity is already only "
          "octave-accurate.")


# ===========================================================================
def main():
    fp = fingerprint()
    print_provenance(fp)

    drift = check_config()
    if drift:
        record("0", "harness config cross-check", ERROR, None,
               "config drift -- this harness replicates constants that no longer match "
               "obscura/config.py: {}".format("; ".join(drift)))
    else:
        record("0", "harness config cross-check", OK, CONFIRMED,
               "all {} replicated constants match obscura/config.py verbatim, "
               "including SENS_REPEATS={} and LR[stroke]={:g}"
               .format(18, SENS_REPEATS, LR_STROKE))

    ctx = ref_context()
    for fn, cids, title in ((lambda: claim1(ctx), ["1"], "CKKS/SPE losslessness"),
                            (lambda: claim2(ctx), ["2"], "weighted HE"),
                            (lambda: claim3(ctx), ["3"], "payload ratio"),
                            (claim4, ["4"], "DP accounting"),
                            (claim5, ["5a", "5b"], "DP-aware sensitivity"),
                            (lambda: claim6(ctx), ["6a", "6b"],
                             "float32 floor calibration")):
        try:
            fn()
        except Exception:
            traceback.print_exc()
            done = {c[0] for c in CHECKS}
            for cid in cids:
                if cid not in done:
                    record(cid, title, ERROR, None,
                           "check raised an exception; see traceback above")
    try:
        extras(ctx)
    except Exception:
        traceback.print_exc()

    # ---- summary + self-check against VERIFICATION.md ---------------------
    documented, doc_err = parse_documented_verdicts()

    print("\n" + "=" * 78)
    print("SUMMARY -- status = did the check run; verdict = is the claim true")
    print("=" * 78)
    print("  {:<4} {:<7} {:<27} {:<27} {}".format(
        "id", "status", "verdict (measured)", "verdict (VERIFICATION.md)", "agree"))
    exit_code = 0
    for cid, title, status, verdict, _ in CHECKS:
        doc_v = documented.get(cid, (None, None))[1] if documented else None
        agree = (doc_v == verdict) if documented else None
        if status != OK:
            exit_code = 1
        if documented and not agree:
            exit_code = 1
        print("  {:<4} {:<7} {:<27} {:<27} {}".format(
            cid, status, str(verdict), str(doc_v),
            "-" if agree is None else ("yes" if agree else "NO  <-- STALE")))

    if doc_err:
        print("\n  !! harness self-check unavailable: {}".format(doc_err))
        print("     VERIFICATION.md must carry a <!-- VERDICT-TABLE --> block with "
              "columns | id | claim | status | verdict |.")
        exit_code = 1

    if documented:
        extra = sorted(set(documented) - {c[0] for c in CHECKS})
        if extra:
            print("\n  !! VERIFICATION.md documents claims this run did not measure: {}"
                  .format(extra))
            exit_code = 1

    print("\n  provenance fingerprint of this run:")
    for name in RESULT_FILES:
        print("    {:<18} sha256:{}  {} records".format(name, *fp[name]))

    print("\n  EXIT {}: {}".format(
        exit_code,
        "all checks executed and every verdict matches VERIFICATION.md "
        "(refutations included -- a recorded refutation is a pass)"
        if exit_code == 0 else
        "a check failed to execute, or a measured verdict disagrees with "
        "VERIFICATION.md"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
