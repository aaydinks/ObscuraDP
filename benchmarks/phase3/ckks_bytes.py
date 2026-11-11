"""Measure how large one update is when it is encrypted with CKKS (research-only backend).

    python -m benchmarks.phase3.ckks_bytes    (needs TenSEAL: the ``he`` or ``legacy`` extra)

The encryption backend is not an arm of the comparison: it is research-only, and what a
model learns under it is what the local arm learns, because the server adds the same
updates either way (up to an error of about 1e-4). Its cost is the size of its messages.
This script encrypts one update of each model with the real wrapper
(``obscuradp.crypto.ckks``) and writes the sizes to ``ckks_bytes.json`` next to the results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.phase2.tables import HIDDEN, SCHEMAS
from benchmarks.phase3.common import MODELS
from obscuradp.crypto.ckks import CkksClient, CkksKeyHolder
from obscuradp.models import TabularMLP
from obscuradp.unsafe import UnsafeFlags

RESULTS = Path(__file__).resolve().parents[1] / "results" / "phase3"


def main() -> None:
    holder = CkksKeyHolder(UnsafeFlags(allow_unsafe=True))
    client = CkksClient("site-0", holder.public_context())
    generator = np.random.default_rng(0)
    sizes = {}
    for dataset in ("stroke", "loan"):
        for model in MODELS:
            hidden = HIDDEN[dataset] if model == "network" else ()
            network = TabularMLP(SCHEMAS[dataset].width, hidden)
            numbers = sum(int(value.numel()) for value in network.parameters())
            update = generator.normal(scale=0.01, size=numbers)
            encrypted = sum(len(part) for part in client.protect(update))
            sizes[f"{dataset}/{model}"] = {
                "numbers": numbers,
                "plain_bytes": 4 * numbers,
                "encrypted_bytes": encrypted,
                "ratio": encrypted / (4 * numbers),
            }
            print(
                f"{dataset}/{model}: {numbers} numbers, {encrypted} bytes, {encrypted / (4 * numbers):.1f} times plain"
            )
    output = {
        "note": "One update encrypted with obscuradp.crypto.ckks, measured from the real "
        "messages. The public context that every participant also needs is not included.",
        "sizes": sizes,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ckks_bytes.json").write_text(json.dumps(output, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
