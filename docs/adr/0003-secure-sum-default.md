# 0003. Default way to hide individual updates from the server

Status: accepted by the owner on 2026-11-10.

## Context

Today a single program holds the decryption key, adds the encrypted updates and decrypts the result
(G5). The encryption scheme in use, CKKS, gives approximate results, and publishing its decrypted values can
expose the key (G15). Its messages are also about 14 times larger than plain ones (G7), while the website
target is to be at least 50% smaller.

Three ways to let a server learn only the sum:

| Option | Exact | Message size against plain | Main weakness |
|---|---|---|---|
| Pairwise masking over integers | Yes | 0.3 to 0.5 with 8 to 12 bit values | A participant dropping out stalls the round unless extra recovery steps are added |
| BFV encryption on integers | Yes | Estimated 2 to 5 times larger, to be measured | Whoever holds the key must refuse to decrypt anything but the final sum |
| CKKS encryption | No | About 14 times larger, measured | Key exposure when decrypted values are published |

## Decision

1. The default is pairwise masking over integers, built on the `cryptography` package, with no dependency on
   TenSEAL.
2. Exact BFV through TenSEAL is the optional encryption backend.
3. CKKS stays available as `research_only`, raises an unsafe flag unless every role is in one trust domain,
   and never hands back 64-bit decrypted values.
4. Roles are separate objects: `KeyHolder`, `ClientEncryptor`, `Aggregator`. A test proves the aggregator
   cannot decrypt.
5. No hand-written noise flooding for CKKS. If it is ever needed, it comes from a library that implements
   it (OpenFHE), as an optional extra.
6. In v0.1 a round in which a participant drops out is retried with fresh randomness. Dropout recovery
   comes later through Flower's SecAgg+.

## Consequences

- The website's "homomorphic encryption" story becomes "a secure sum, with encryption as one option". The
  copy needs to follow (S22).
- The bytes target becomes reachable for the default backend only, and the site should say so.
- Selective encryption (SPE) only applies to the encryption backends and only in local mode.
