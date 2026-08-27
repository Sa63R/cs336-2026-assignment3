from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import typer
from numpy.lib.format import open_memmap
from rich.console import Console

from cs336_scaling.local.integrity import sha256_file
from cs336_scaling.local.schemas import DatasetManifest
from cs336_scaling.local.settings import LocalSettings


app = typer.Typer(help="Prepare deterministic local token datasets.")
console = Console()


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def dataset_id_for(
    *,
    source: str,
    source_revision: str | None,
    tokenizer: str,
    tokenizer_revision: str | None,
    seed: int,
    train_sha256: str,
    validation_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "source": source,
            "source_revision": source_revision,
            "tokenizer": tokenizer,
            "tokenizer_revision": tokenizer_revision,
            "seed": seed,
            "train_sha256": train_sha256,
            "validation_sha256": validation_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.blake2s(payload, digest_size=16).hexdigest()


def require_new_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing dataset manifest: {manifest_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def finalize_array(partial_path: Path, final_path: Path, array: np.memmap) -> None:
    array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()
    partial_path.replace(final_path)


def write_manifest(
    output_dir: Path,
    *,
    source: str,
    source_revision: str | None,
    tokenizer: str,
    tokenizer_revision: str | None,
    vocab_size: int,
    eos_token_id: int,
    seed: int,
    train_path: Path,
    validation_path: Path,
) -> Path:
    train_hash = sha256_file(train_path)
    validation_hash = sha256_file(validation_path)
    manifest = DatasetManifest(
        dataset_id=dataset_id_for(
            source=source,
            source_revision=source_revision,
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            train_sha256=train_hash,
            validation_sha256=validation_hash,
        ),
        source=source,
        source_revision=source_revision,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        seed=seed,
        train_tokens=int(np.load(train_path, mmap_mode="r").size),
        validation_tokens=int(np.load(validation_path, mmap_mode="r").size),
        train_tokens_path=Path(train_path.name),
        validation_tokens_path=Path(validation_path.name),
        train_sha256=train_hash,
        validation_sha256=validation_hash,
        created_at=utc_now(),
    )
    manifest_path = output_dir / "manifest.json"
    temporary_path = manifest_path.with_suffix(".json.partial")
    temporary_path.write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def fill_synthetic_array(
    path: Path,
    *,
    tokens: int,
    vocab_size: int,
    seed: int,
) -> None:
    partial_path = path.with_suffix(".partial.npy")
    array = open_memmap(partial_path, mode="w+", dtype=np.uint16, shape=(tokens,))
    rng = np.random.default_rng(seed)
    chunk_size = 1_000_000
    for start in range(0, tokens, chunk_size):
        stop = min(start + chunk_size, tokens)
        array[start:stop] = rng.integers(
            0, vocab_size, size=stop - start, dtype=np.uint16
        )
    finalize_array(partial_path, path, array)


@app.command()
def synthetic(
    name: str = typer.Option("smoke"),
    train_tokens: int = typer.Option(512 * 64 + 1, min=513),
    validation_tokens: int = typer.Option(512 * 8 + 1, min=513),
    vocab_size: int = typer.Option(256, min=2, max=65_536),
    seed: int = typer.Option(67, min=0),
) -> None:
    """Create deterministic random tokens for infrastructure smoke tests."""

    settings = LocalSettings.from_env()
    output_dir = require_new_output_dir(settings.datasets_dir / name)
    train_path = output_dir / "train.npy"
    validation_path = output_dir / "validation.npy"
    fill_synthetic_array(
        train_path, tokens=train_tokens, vocab_size=vocab_size, seed=seed
    )
    fill_synthetic_array(
        validation_path,
        tokens=validation_tokens,
        vocab_size=vocab_size,
        seed=seed + 1,
    )
    manifest = write_manifest(
        output_dir,
        source="synthetic-random",
        source_revision=None,
        tokenizer="synthetic-token-ids",
        tokenizer_revision=None,
        vocab_size=vocab_size,
        eos_token_id=0,
        seed=seed,
        train_path=train_path,
        validation_path=validation_path,
    )
    console.print(f"Wrote synthetic dataset manifest: [bold]{manifest}[/bold]")


def fill_from_text_stream(
    arrays: list[np.memmap],
    texts: Iterable[list[str]],
    *,
    tokenizer,
    eos_token_id: int,
) -> None:
    array_index = 0
    write_position = 0
    for batch in texts:
        tokenized = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for document_tokens in tokenized:
            values = np.asarray([*document_tokens, eos_token_id], dtype=np.uint16)
            while values.size and array_index < len(arrays):
                array = arrays[array_index]
                available = array.size - write_position
                take = min(available, values.size)
                array[write_position : write_position + take] = values[:take]
                write_position += take
                values = values[take:]
                if write_position == array.size:
                    array_index += 1
                    write_position = 0
            if array_index == len(arrays):
                return
    raise RuntimeError("the streamed dataset ended before the requested token count")


@app.command()
def dclm(
    name: str = typer.Option("dclm-local"),
    train_tokens: int = typer.Option(10_000_001, min=513),
    validation_tokens: int = typer.Option(262_145, min=513),
    seed: int = typer.Option(67, min=0),
    dataset_name: str = typer.Option("mlfoundations/dclm-baseline-1.0-parquet"),
    dataset_revision: str = typer.Option("main"),
    tokenizer_name: str = typer.Option("NousResearch/Llama-2-7b-hf"),
    tokenizer_revision: str = typer.Option("main"),
    text_batch_size: int = typer.Option(128, min=1, max=4_096),
) -> None:
    """Stream, tokenize, and persist a deterministic DCLM subset."""

    import pyarrow.parquet as parquet
    from huggingface_hub import HfApi, HfFileSystem
    from transformers import AutoTokenizer

    settings = LocalSettings.from_env()
    output_dir = require_new_output_dir(settings.datasets_dir / name)
    train_path = output_dir / "train.npy"
    validation_path = output_dir / "validation.npy"
    train_partial = train_path.with_suffix(".partial.npy")
    validation_partial = validation_path.with_suffix(".partial.npy")
    train_array = open_memmap(
        train_partial, mode="w+", dtype=np.uint16, shape=(train_tokens,)
    )
    validation_array = open_memmap(
        validation_partial,
        mode="w+",
        dtype=np.uint16,
        shape=(validation_tokens,),
    )

    hub_api = HfApi()
    dataset_info = hub_api.dataset_info(dataset_name, revision=dataset_revision)
    resolved_dataset_revision = dataset_info.sha
    if not resolved_dataset_revision:
        raise RuntimeError(f"could not resolve dataset revision for {dataset_name!r}")
    parquet_files = sorted(
        path
        for path in hub_api.list_repo_files(
            dataset_name, repo_type="dataset", revision=resolved_dataset_revision
        )
        if path.endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(f"dataset {dataset_name!r} contains no Parquet files")
    parquet_files = random.Random(seed).sample(parquet_files, len(parquet_files))
    tokenizer_info = hub_api.model_info(tokenizer_name, revision=tokenizer_revision)
    resolved_tokenizer_revision = tokenizer_info.sha
    if not resolved_tokenizer_revision:
        raise RuntimeError(
            f"could not resolve tokenizer revision for {tokenizer_name!r}"
        )
    tokenizer = cast(
        Any,
        AutoTokenizer.from_pretrained(
            tokenizer_name, revision=resolved_tokenizer_revision, use_fast=True
        ),
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError(f"tokenizer {tokenizer_name!r} has no EOS token")
    if tokenizer.vocab_size > 65_536:
        raise ValueError("the tokenizer vocabulary does not fit in uint16")
    filesystem = HfFileSystem()

    def parquet_batches() -> Iterator[list[str]]:
        for file_index, repo_path in enumerate(parquet_files, start=1):
            console.print(
                f"Reading Parquet shard {file_index}/{len(parquet_files)}: {repo_path}"
            )
            remote_path = (
                f"datasets/{dataset_name}@{resolved_dataset_revision}/{repo_path}"
            )
            with filesystem.open(remote_path, "rb") as remote_file:
                parquet_file = parquet.ParquetFile(remote_file)
                for record_batch in parquet_file.iter_batches(
                    batch_size=text_batch_size, columns=["text"]
                ):
                    yield [
                        text
                        for text in record_batch.column("text").to_pylist()
                        if isinstance(text, str) and text
                    ]

    console.print(
        f"Streaming {train_tokens + validation_tokens:,} tokens from "
        f"{dataset_name}@{resolved_dataset_revision[:12]}"
    )
    fill_from_text_stream(
        [train_array, validation_array],
        parquet_batches(),
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
    )
    finalize_array(train_partial, train_path, train_array)
    finalize_array(validation_partial, validation_path, validation_array)
    manifest = write_manifest(
        output_dir,
        source=dataset_name,
        source_revision=resolved_dataset_revision,
        tokenizer=tokenizer_name,
        tokenizer_revision=resolved_tokenizer_revision,
        vocab_size=tokenizer.vocab_size,
        eos_token_id=eos_token_id,
        seed=seed,
        train_path=train_path,
        validation_path=validation_path,
    )
    console.print(f"Wrote DCLM dataset manifest: [bold]{manifest}[/bold]")


if __name__ == "__main__":
    app()
