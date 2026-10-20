r"""Benchmark rows for the research agent.

Two benchmarks, one row type.

**BrowseComp** ships as a CSV published by OpenAI simple-evals. Its ``problem``
and ``answer`` columns are base64-encoded and XOR-encrypted, so the questions do
not leak into training corpora as plain text. Each row carries its own
``canary`` column, and that canary is the key material for that row. Decryption
is obfuscation, not security.

**FRAMES** ships as plain text on the Hugging Face hub, with no encryption.

**The golden set** is ``golden.jsonl`` beside this module, spec.md section 8.5.
Each line is a FRAMES question with a checklist of facts a report must state,
so it gets its own row type and its own judge.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import random
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

BENCHMARKS = ("browsecomp", "frames")
GOLDEN = "golden"
GOLDEN_PATH = Path(__file__).with_name("golden.jsonl")

BROWSECOMP_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)

FRAMES_DATASET = "google/frames-benchmark"
FRAMES_CONFIG = "default"
FRAMES_SPLIT = "test"
FRAMES_ROWS_URL = "https://datasets-server.huggingface.co/rows"

# The rows endpoint refuses a larger page, and the split is under a thousand
# rows, so a whole benchmark is a handful of requests.
FRAMES_PAGE_SIZE = 100
FRAMES_MAX_ROWS = 5000


@dataclass(frozen=True)
class Row:
    """One benchmark question with its gold answer."""

    question_id: str
    question: str
    answer: str


@dataclass(frozen=True)
class GoldenRow:
    """One golden question with the checklist its report is judged against."""

    question_id: str
    question: str
    required_facts: list[str]
    wrong_claims: list[str]


NonBlank = Annotated[str, Field(pattern=r"\S")]


class GoldenLine(BaseModel):
    """One line of ``golden.jsonl``, exactly as spec.md section 8.5 names it."""

    model_config = ConfigDict(extra="forbid")

    id: NonBlank
    question: NonBlank
    difficulty: Literal["easy", "medium", "hard"]
    hops: int = Field(ge=1)
    required_facts: list[NonBlank] = Field(min_length=1)
    wrong_claims: list[NonBlank]
    sources: list[NonBlank] = Field(min_length=1)
    frames_row: int = Field(ge=0)


_GOLDEN_LINE = TypeAdapter(GoldenLine)


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenRow]:
    """Load every golden question, refusing a file with any bad line.

    A malformed line is fatal, unlike a results file. A golden line that
    quietly dropped out would raise the pass rate of every run after it.

    Args:
        path: The golden JSON lines file.

    Returns:
        Every golden question, in file order.

    Raises:
        ValueError: If a line does not match :class:`GoldenLine`, or two lines
            share an id.
    """
    rows: list[GoldenRow] = []
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            line = _GOLDEN_LINE.validate_json(text)
        except ValueError as error:
            raise ValueError(f"{path.name} line {number}: {error}") from error
        rows.append(
            GoldenRow(
                question_id=line.id,
                question=line.question,
                required_facts=line.required_facts,
                wrong_claims=line.wrong_claims,
            )
        )

    counts = Counter(row.question_id for row in rows)
    repeated = sorted(question_id for question_id, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(f"{path.name} repeats ids {json.dumps(repeated)}")

    logger.info(f"golden: loaded {len(rows)} rows")
    return rows


def derive_key(password: str, length: int) -> bytes:
    """Derive a repeating XOR key of ``length`` bytes from ``password``.

    Args:
        password: The row's canary string.
        length: How many key bytes are needed.

    Returns:
        Exactly ``length`` bytes of key material.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt_string(ciphertext_b64: str, password: str) -> str:
    """Decrypt one base64-encoded, XOR-encrypted field.

    Args:
        ciphertext_b64: The encrypted field as stored in the CSV.
        password: The canary string used as key material.

    Returns:
        The decrypted text.
    """
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8", errors="replace")


def fetch_bytes(url: str, timeout: float) -> bytes:
    """Fetch one URL and return its body.

    Args:
        url: The address to fetch.
        timeout: Socket timeout in seconds.

    Returns:
        The response body.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def load_browsecomp(timeout: float) -> list[Row]:
    """Load and decrypt every BrowseComp question.

    Args:
        timeout: Socket timeout in seconds.

    Returns:
        Every row in the published test set, in file order.
    """
    body = fetch_bytes(BROWSECOMP_CSV_URL, timeout).decode("utf-8")
    reader = csv.DictReader(io.StringIO(body))

    rows: list[Row] = []
    for index, raw in enumerate(reader):
        canary = raw.get("canary") or ""
        rows.append(
            Row(
                question_id=f"browsecomp-{index}",
                question=decrypt_string(raw["problem"], canary),
                answer=decrypt_string(raw["answer"], canary),
            )
        )

    logger.info(f"browsecomp: loaded {len(rows)} rows")
    return rows


def load_frames(timeout: float) -> list[Row]:
    """Load every FRAMES question from the Hugging Face rows endpoint.

    Args:
        timeout: Socket timeout in seconds, applied to each page request.

    Returns:
        Every row in the test split, in dataset order.
    """
    rows: list[Row] = []
    offset = 0

    while offset < FRAMES_MAX_ROWS:
        params = urllib.parse.urlencode(
            {
                "dataset": FRAMES_DATASET,
                "config": FRAMES_CONFIG,
                "split": FRAMES_SPLIT,
                "offset": offset,
                "length": FRAMES_PAGE_SIZE,
            }
        )
        payload = json.loads(fetch_bytes(f"{FRAMES_ROWS_URL}?{params}", timeout))
        page = payload.get("rows", [])
        if not page:
            break

        for entry in page:
            raw = entry["row"]
            index = entry.get("row_idx", offset)
            rows.append(
                Row(
                    question_id=f"frames-{index}",
                    question=raw["Prompt"],
                    answer=str(raw["Answer"]),
                )
            )
        offset += len(page)

    logger.info(f"frames: loaded {len(rows)} rows")
    return rows


def subset(rows: list[Row], subset_size: int, seed: int, offset: int) -> list[Row]:
    """Pick a reproducible subset of the loaded rows.

    The seed alone fixes the shuffled order, so the same seed always yields the
    same sequence. ``offset`` slides a window along that fixed order, which lets
    a second run take fresh questions without changing the shuffle.

    Args:
        rows: Every row the benchmark published.
        subset_size: How many questions to return.
        seed: Seed for the shuffle.
        offset: Where the window starts in the shuffled order.

    Returns:
        The chosen rows.

    Raises:
        ValueError: If the window falls outside the loaded rows.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    chosen = shuffled[offset : offset + subset_size]
    if len(chosen) < subset_size:
        raise ValueError(
            f"asked for {subset_size} rows at offset {offset}, but only "
            f"{len(chosen)} of {len(rows)} are available"
        )
    return chosen


def load_subset(
    benchmark: str, subset_size: int, seed: int, offset: int, timeout: float
) -> list[Row]:
    """Load one benchmark and return its reproducible subset.

    Args:
        benchmark: One of :data:`BENCHMARKS`.
        subset_size: How many questions to return.
        seed: Seed for the shuffle.
        offset: Where the window starts in the shuffled order.
        timeout: Socket timeout in seconds.

    Returns:
        The chosen rows.

    Raises:
        ValueError: If the benchmark name is unknown.
    """
    if benchmark == "browsecomp":
        rows = load_browsecomp(timeout)
    elif benchmark == "frames":
        rows = load_frames(timeout)
    else:
        raise ValueError(f"Unknown benchmark {benchmark!r}. Expected one of {BENCHMARKS}.")

    return subset(rows, subset_size, seed, offset)
