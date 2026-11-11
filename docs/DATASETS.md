# The two dataset files in this repository

Written in step S12 (2026-11-11) after the owner's decision that the files stay in the repository with a
note on where they come from. This note lives outside the research folders because those are frozen.

**No dataset is part of the `obscuradp` package.** `tools/check_release_contents.py` fails a release that
contains any `.csv` file or any folder named `data`, and it runs on every pull request.

## What is here

| | Stroke table | Loan table |
|---|---|---|
| File | `Experiment 1/data/healthcare-dataset-stroke-data.csv` | `Experiment 2/data/loan_approval_dataset.csv` |
| Second copy (byte for byte the same) | `Trial 07/27/data/healthcare-dataset-stroke-data.csv` | `Trial 07/27/data/loan_approval_dataset.csv` |
| Size | 316,971 bytes | 384,337 bytes |
| Rows (without the header) | 5,110 | 4,269 |
| Columns | 12 | 13 |
| Line endings | LF, no final line break | CRLF |
| sha256 of the file as stored | `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420` | `4b5cd093d178378f4cfa8c107adb6e599b88be9d8a3b51f3b99c0d5914154e54` |
| Used by | Experiment 1, Trial 07/27, the Phase 2 runs (`benchmarks/phase2/tables.py`) | Experiment 2, Trial 07/27, the Phase 2 runs |

The fingerprint list of the frozen folders (`tests/golden/research_manifest.json`) shows a different value
for the loan file (`7fd55001...`). That list converts CRLF line endings to LF before hashing, so that a
checkout on Windows gives the same result. The file itself has not changed.

## Where the files come from

What the research notes in this repository say:

| | Stroke table | Loan table |
|---|---|---|
| Named source | "fedesoriano Stroke Prediction" on Kaggle (`Experiment 1/README.md:18`, `Experiment 1/REPRODUCTION.md:12`) | Kaggle `architsharma01/loan-approval-prediction` (`Experiment 2/README.md:17`) |
| How it was fetched | "from a public GitHub mirror (Kaggle requires auth)" (`Experiment 1/REPRODUCTION.md:74`) | "mirror used because Kaggle needs auth" (`Experiment 2/README.md:138`) |
| Which mirror | **not found**: no address was recorded | **not found**: no address was recorded |
| Source page to check | `https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset` | `https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset` |

What was checked on 2026-11-11 and what was not:

- The row counts above equal the ones the research notes give for the Kaggle tables (5,110 and 4,269). The
  loan file has the 11 feature columns those notes mention, plus an identifier and the outcome.
- The two source pages could **not** be opened from the working session (the host is outside the network
  rules of the cloud environment). The page addresses above are the usual addresses of those two Kaggle
  entries and were not confirmed by opening them.
- Because the mirror is unknown and the pages were not opened, it is **not confirmed** that the files here
  are the same, byte for byte, as the files on Kaggle.

## License

| | Stroke table | Loan table |
|---|---|---|
| License | **not found** | **not found** |

A web search on 2026-11-11 did not show the license field of either Kaggle page, and other sites that copy
the tables give no dependable statement. Nothing is assumed here. Whoever can open the two pages while
logged in can read the license field there and replace "not found" in this note.

Until then:

- Treat both files as third-party data with unknown terms. They are kept only so that the research results
  in this repository can be reproduced.
- Do not copy them into other projects on the strength of this repository's own license (Apache-2.0). That
  license covers the code and documents, not these two files.
- The benchmark of step S17 downloads its datasets from their sources with checksums and does not rely on
  these copies.
