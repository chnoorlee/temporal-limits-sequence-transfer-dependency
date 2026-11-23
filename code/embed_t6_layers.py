#!/usr/bin/env python3
"""Compute residue-chunked ESM-2 t6 embeddings at intermediate layers.

Used only for the post-review layer-selection probe. Pooling follows
``03_embed_proteins.py`` exactly (inverse-coverage-weighted residue mean over
overlapping windows), so layer comparisons differ only in the hidden layer.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_embedder():
    path = Path(__file__).resolve().parent / "03_embed_proteins.py"
    spec = importlib.util.spec_from_file_location("embed_proteins", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--layers", default="2,4,6")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    embedder = _load_embedder()
    import torch

    layers = [int(value) for value in args.layers.split(",")]
    records = embedder.load_sequences(args.fasta)
    snapshot = embedder.resolve_snapshot(local_files_only=False)
    tokenizer, model = embedder.load_pinned_model(snapshot, device=args.device)
    model.config.output_hidden_states = True

    total_chunks = embedder._total_chunks(
        records, strategy="chunked", chunk_size=embedder.MAX_RESIDUES,
        overlap=embedder.DEFAULT_OVERLAP,
    )
    accumulators = None
    denominators = np.zeros(len(records), dtype=np.float64)
    completed = 0
    model.eval()
    with torch.inference_mode():
        for chunk_batch in embedder._batched(
            embedder.iter_chunks(
                records, strategy="chunked", chunk_size=embedder.MAX_RESIDUES,
                overlap=embedder.DEFAULT_OVERLAP,
            ),
            args.batch_size,
        ):
            encoded = tokenizer(
                [chunk.fragment for chunk in chunk_batch],
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_tokens_mask = encoded.pop("special_tokens_mask")
            model_inputs = {key: value.to(args.device) for key, value in encoded.items()}
            special_tokens_mask = special_tokens_mask.to(args.device)
            output = model(**model_inputs, output_hidden_states=True)
            residue_mask = model_inputs["attention_mask"].bool() & ~special_tokens_mask.bool()
            hidden_states = output.hidden_states
            if accumulators is None:
                accumulators = {
                    layer: np.zeros((len(records), int(hidden_states[layer].shape[-1])), dtype=np.float64)
                    for layer in layers
                }
            for row, chunk in enumerate(chunk_batch):
                positions = torch.nonzero(residue_mask[row], as_tuple=False).flatten()
                if int(positions.numel()) != len(chunk.fragment):
                    raise RuntimeError("residue-token count mismatch")
                weights = chunk.residue_weights[:, None]
                for layer in layers:
                    hidden = hidden_states[layer][row, positions].detach().float().cpu().numpy()
                    accumulators[layer][chunk.sequence_index] += np.sum(hidden * weights, axis=0)
                denominators[chunk.sequence_index] += float(np.sum(chunk.residue_weights))
            completed += len(chunk_batch)
            print(f"[layers] chunks {completed}/{total_chunks}", flush=True)

    names = [record.name for record in records]
    (args.outdir / "esm2_t6_8M_layers.names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    for layer in layers:
        values = accumulators[layer] / denominators[:, None]
        path = args.outdir / f"esm2_t6_8M_layer{layer}.npy"
        np.save(path, values.astype(np.float32))
        print("wrote", path, values.shape)


if __name__ == "__main__":
    main()
