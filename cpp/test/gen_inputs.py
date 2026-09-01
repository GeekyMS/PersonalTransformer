"""Generates fixed-seed random inputs for an op's contract.json entry, written
as raw float32 .bin files the C++ test binary loads via load_binary(). See
README.md for the full protocol.

Usage: python gen_inputs.py <op_name> [--dump-dir cpp/test/dumps] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np

CONTRACT_PATH = Path(__file__).parent / "contract.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("op_name")
    parser.add_argument("--dump-dir", default="cpp/test/dumps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    contract = json.loads(CONTRACT_PATH.read_text())
    entry = next(e for e in contract["ops"] if e["op"] == args.op_name)

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for name, shape in entry["inputs"].items():
        arr = rng.normal(0, 1, shape).astype(np.float32)
        arr.tofile(dump_dir / f"{args.op_name}_input_{name}.bin")
        print(f"wrote {name} {shape} -> {args.op_name}_input_{name}.bin")


if __name__ == "__main__":
    main()
