"""The benchmark's datasets beyond the two research tables: their sources and how they are read.

Two public datasets with a natural split into participants, both as packaged by the
TensorFlow Federated project (Apache License 2.0 for the packaging):

- **Handwriting (Federated EMNIST, "FEMNIST")**: 3,400 writers, 671,585 training images
  and 77,483 test images of 28 by 28 grey pixels, 62 classes (digits and letters). The
  images come from NIST Special Database 19, which is in the public domain. A participant
  is a writer.
- **Text (Shakespeare)**: 715 speaking roles of the plays, about 2.4 million characters.
  The plays are in the public domain. A participant is a role, and the task is to say
  which character comes next.

Each archive is downloaded once (``fetch.py``), checked against its recorded fingerprint,
and converted into plain number files next to it. Those files are then mapped into memory,
so several runs on one machine share them. Nothing here is stored in the repository.

How the rows are cut: a participant's rows from the archive's training file are its
training rows, except the last tenth, which are its validation rows; its rows from the
archive's test file are its test rows. The cut does not depend on a seed.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from benchmarks.phase5.fetch import Source, cache_directory, fetch
from obscuradp.tables import ClientData

TFF = "https://storage.googleapis.com/tff-datasets-public"
FEMNIST = Source(
    name="fed_emnist.tar.bz2",
    url=f"{TFF}/fed_emnist.tar.bz2",
    sha256="fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0",
    size=169_808_360,
    origin=(
        "Federated EMNIST as packaged by TensorFlow Federated (Apache-2.0); images from "
        "NIST Special Database 19 (public domain)"
    ),
)
SHAKESPEARE = Source(
    name="shakespeare.tar.bz2",
    url=f"{TFF}/shakespeare.tar.bz2",
    sha256="0285be9906cb5f268092eee4edeeacfc2af4574f2941f7cc2f08a321d7f5c707",
    size=1_848_122,
    origin=(
        "The Shakespeare dataset as packaged by TensorFlow Federated (Apache-2.0); the "
        "plays are in the public domain"
    ),
)

FEMNIST_CLASSES = 62
FEMNIST_SIDE = 28
VALIDATION_SHARE = 0.1

# Every symbol the text model knows, declared in advance. Position 0 stands for anything else.
SYMBOLS = "\n !\"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
TEXT_VOCABULARY = len(SYMBOLS) + 1
TEXT_WINDOW = 80  # symbols the model reads before it says what comes next
TEXT_STRIDE = 4  # a new window starts every this many symbols
TEXT_CAP = 1_000  # at most this many windows per role and file, the first ones


def _unpack(source: Source, members: tuple[str, ...], folder: Path) -> list[Path]:
    """Take exactly the named files out of a downloaded archive (once)."""
    folder.mkdir(parents=True, exist_ok=True)
    wanted = [folder / member for member in members]
    if all(path.is_file() for path in wanted):
        return wanted
    archive = fetch(source)
    with tarfile.open(archive, "r:bz2") as handle:
        for member in members:
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"{source.name} has no file {member}")
            scratch = folder / (member + ".part")
            with scratch.open("wb") as out:
                while chunk := stream.read(1 << 20):
                    out.write(chunk)
            scratch.replace(folder / member)
    return wanted


def _h5py():  # type: ignore[no-untyped-def]
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on the installation
        raise ImportError(
            "reading the archives needs h5py; install the benchmark extra: "
            "pip install 'obscuradp[bench]'"
        ) from error
    return h5py


def _groups(lengths: list[int]) -> np.ndarray:
    """Where each participant's rows start in the stacked files, plus the end."""
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def prepare_femnist(directory: Path | None = None) -> Path:
    """Convert the handwriting archive into number files (once). Returns their folder.

    Images are stored as rows of 784 numbers with 1 for ink and 0 for paper (the archive
    has it the other way round), writers in the archive's order.
    """
    folder = (directory or cache_directory()) / "femnist"
    if (folder / "done").is_file():
        return folder
    h5py = _h5py()
    files = _unpack(FEMNIST, ("fed_emnist_train.h5", "fed_emnist_test.h5"), folder)
    writers: list[str] | None = None
    for part, path in zip(("train", "test"), files, strict=True):
        with h5py.File(path, "r") as handle:
            examples = handle["examples"]
            names = sorted(examples.keys())
            if writers is None:
                writers = names
            elif names != writers:
                raise ValueError("the two files of the archive do not list the same writers")
            lengths = [int(examples[name]["label"].shape[0]) for name in names]
            starts = _groups(lengths)
            pixels = np.lib.format.open_memmap(
                folder / f"{part}_x.npy",
                mode="w+",
                dtype=np.float32,
                shape=(int(starts[-1]), FEMNIST_SIDE * FEMNIST_SIDE),
            )
            labels = np.zeros(int(starts[-1]), dtype=np.float32)
            for index, name in enumerate(names):
                begin, end = int(starts[index]), int(starts[index + 1])
                pixels[begin:end] = 1.0 - examples[name]["pixels"][()].reshape(end - begin, -1)
                labels[begin:end] = examples[name]["label"][()]
            pixels.flush()
            del pixels
            np.save(folder / f"{part}_y.npy", labels)
            np.save(folder / f"{part}_starts.npy", starts)
    (folder / "writers.txt").write_text("\n".join(writers or []) + "\n")
    (folder / "done").write_text("ok\n")
    return folder


def encode_text(text: str) -> np.ndarray:
    """A text as symbol numbers (0 for a symbol that is not on the declared list)."""
    position = {symbol: index + 1 for index, symbol in enumerate(SYMBOLS)}
    return np.array([position.get(symbol, 0) for symbol in text], dtype=np.float32)


def windows(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut a role's symbols into reading windows and the symbol that follows each."""
    count = max(0, (symbols.size - TEXT_WINDOW - 1) // TEXT_STRIDE + 1)
    count = min(count, TEXT_CAP) if symbols.size > TEXT_WINDOW else 0
    read = np.zeros((count, TEXT_WINDOW), dtype=np.float32)
    following = np.zeros(count, dtype=np.float32)
    for index in range(count):
        begin = index * TEXT_STRIDE
        read[index] = symbols[begin : begin + TEXT_WINDOW]
        following[index] = symbols[begin + TEXT_WINDOW]
    return read, following


def prepare_shakespeare(directory: Path | None = None) -> Path:
    """Convert the text archive into number files (once). Returns their folder."""
    folder = (directory or cache_directory()) / "shakespeare"
    if (folder / "done").is_file():
        return folder
    h5py = _h5py()
    files = _unpack(SHAKESPEARE, ("shakespeare_train.h5", "shakespeare_test.h5"), folder)
    roles: list[str] | None = None
    for part, path in zip(("train", "test"), files, strict=True):
        with h5py.File(path, "r") as handle:
            examples = handle["examples"]
            names = sorted(examples.keys())
            if roles is None:
                roles = names
            elif names != roles:
                raise ValueError("the two files of the archive do not list the same roles")
            read, following, lengths = [], [], []
            for name in names:
                text = "\n".join(
                    snippet.decode("utf-8", "replace") for snippet in examples[name]["snippets"][()]
                )
                x, y = windows(encode_text(text))
                read.append(x)
                following.append(y)
                lengths.append(int(y.size))
            np.save(folder / f"{part}_x.npy", np.concatenate(read))
            np.save(folder / f"{part}_y.npy", np.concatenate(following))
            np.save(folder / f"{part}_starts.npy", _groups(lengths))
    (folder / "roles.txt").write_text("\n".join(roles or []) + "\n")
    (folder / "done").write_text("ok\n")
    return folder


def participants(folder: Path, prefix: str, *, least_training_rows: int = 2) -> list[ClientData]:
    """The participants of a prepared dataset, their rows mapped into memory.

    A participant with fewer than ``least_training_rows`` training rows is left out (it
    could not keep one row for training and one for validation).
    """
    parts = {}
    for part in ("train", "test"):
        parts[part] = (
            np.load(folder / f"{part}_x.npy", mmap_mode="r"),
            np.load(folder / f"{part}_y.npy", mmap_mode="r"),
            np.load(folder / f"{part}_starts.npy"),
        )
    train_x, train_y, train_starts = parts["train"]
    test_x, test_y, test_starts = parts["test"]
    clients = []
    for index in range(train_starts.size - 1):
        begin, end = int(train_starts[index]), int(train_starts[index + 1])
        if end - begin < least_training_rows:
            continue
        held = max(1, round(VALIDATION_SHARE * (end - begin)))
        cut = end - held
        test_begin, test_end = int(test_starts[index]), int(test_starts[index + 1])
        clients.append(
            ClientData(
                name=f"{prefix}-{index:04d}",
                train_x=train_x[begin:cut],
                train_y=train_y[begin:cut],
                validation_x=train_x[cut:end],
                validation_y=train_y[cut:end],
                test_x=test_x[test_begin:test_end],
                test_y=test_y[test_begin:test_end],
            )
        )
    return clients


def load_femnist(directory: Path | None = None) -> list[ClientData]:
    """The writers of the handwriting dataset as participants."""
    return participants(prepare_femnist(directory), "writer")


def load_shakespeare(directory: Path | None = None) -> list[ClientData]:
    """The speaking roles of the text dataset as participants."""
    return participants(prepare_shakespeare(directory), "role")
