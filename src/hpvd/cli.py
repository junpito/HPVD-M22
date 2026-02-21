"""
HPVD CLI — lightweight command-line interface
==============================================

Usage::

    # Build index from a folder of HPVDInputBundle JSON files
    python -m src.hpvd.cli build-index --bundles data/bundles/ --output artifacts/

    # Search using a query bundle (JSON file or stdin)
    python -m src.hpvd.cli search --index artifacts/ --query query.json
    echo '{ ... }' | python -m src.hpvd.cli search --index artifacts/

Output is ``hpvd_output_v1`` JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from .engine import HPVDEngine, HPVDConfig, HPVD_Output
from .trajectory import HPVDInputBundle


# ------------------------------------------------------------------
# Bundle I/O helpers
# ------------------------------------------------------------------

def _bundle_from_dict(d: Dict) -> HPVDInputBundle:
    """Reconstruct an ``HPVDInputBundle`` from a JSON-compatible dict.

    Expected keys:
        trajectory  – 2-D list (rows × cols) **or** path to ``.npy`` file
        dna         – 1-D list **or** path to ``.npy`` file
        geometry_context  – dict
        metadata    – dict
    """
    # trajectory
    traj = d["trajectory"]
    if isinstance(traj, str):
        traj = np.load(traj)
    else:
        traj = np.asarray(traj, dtype=np.float32)

    # dna
    dna = d["dna"]
    if isinstance(dna, str):
        dna = np.load(dna)
    else:
        dna = np.asarray(dna, dtype=np.float32)

    return HPVDInputBundle(
        trajectory=traj.astype(np.float32),
        dna=dna.astype(np.float32),
        geometry_context=d.get("geometry_context", {}),
        metadata=d.get("metadata", {}),
    )


def _load_bundles_from_folder(folder: str) -> List[HPVDInputBundle]:
    """Load all ``*.json`` bundle files from *folder*."""
    bundles: List[HPVDInputBundle] = []
    folder_path = Path(folder)
    for fp in sorted(folder_path.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support single bundle or list of bundles
        if isinstance(data, list):
            bundles.extend(_bundle_from_dict(b) for b in data)
        else:
            bundles.append(_bundle_from_dict(data))
    return bundles


def _load_bundles_from_file(path: str) -> List[HPVDInputBundle]:
    """Load a single JSON file that contains one or a list of bundles."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [_bundle_from_dict(b) for b in data]
    return [_bundle_from_dict(data)]


# ------------------------------------------------------------------
# Subcommands
# ------------------------------------------------------------------

def cmd_build_index(args: argparse.Namespace) -> None:
    """``build-index`` subcommand."""
    # Gather bundles
    bundles: List[HPVDInputBundle] = []
    src = args.bundles
    if os.path.isdir(src):
        bundles = _load_bundles_from_folder(src)
    elif os.path.isfile(src):
        bundles = _load_bundles_from_file(src)
    else:
        print(f"Error: {src} is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    if not bundles:
        print("Error: no bundles found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(bundles)} bundles.", file=sys.stderr)

    engine = HPVDEngine(HPVDConfig())
    engine.build_from_bundles(bundles)

    out_dir = args.output
    engine.save(out_dir)
    print(f"Index saved to {out_dir}", file=sys.stderr)


def cmd_search(args: argparse.Namespace) -> None:
    """``search`` subcommand."""
    # Load engine artifacts
    engine = HPVDEngine(HPVDConfig())
    engine.load(args.index)

    # Load query bundle (file or stdin)
    if args.query and args.query != "-":
        with open(args.query, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.load(sys.stdin)

    query_bundle = _bundle_from_dict(raw)
    query_bundle.validate()

    output: HPVD_Output = engine.search_families(query_bundle)

    # Write hpvd_output_v1 JSON to stdout
    print(output.to_json(indent=2))


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpvd",
        description="HPVD — Hybrid Probabilistic Vector Database CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-index
    p_build = sub.add_parser(
        "build-index",
        help="Build HPVD index from HPVDInputBundle JSON files",
    )
    p_build.add_argument(
        "--bundles",
        required=True,
        help="Path to a folder of JSON bundle files, or a single JSON file",
    )
    p_build.add_argument(
        "--output",
        required=True,
        help="Directory to save index artifacts (PCA, FAISS, config)",
    )

    # search
    p_search = sub.add_parser(
        "search",
        help="Search for analog families given a query bundle",
    )
    p_search.add_argument(
        "--index",
        required=True,
        help="Directory containing saved HPVD artifacts",
    )
    p_search.add_argument(
        "--query",
        default="-",
        help="Path to query bundle JSON file (default: read from stdin)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-index":
        cmd_build_index(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
