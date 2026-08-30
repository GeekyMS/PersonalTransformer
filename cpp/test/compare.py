"""Op-by-op correctness check: load a C++ dump, load the matching NumPy
intermediate, np.allclose them. See contract.json for the shape/op registry
and README.md for the full protocol.

Usage: python compare.py <op_name> [--dump-dir cpp/test/dumps]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

CONTRACT_PATH = Path(__file__).parent / "contract.json"


def load_contract(op_name: str) -> dict:
    contract = json.loads(CONTRACT_PATH.read_text())
    for entry in contract["ops"]:
        if entry["op"] == op_name:
            return entry
    raise KeyError(
        f"No contract entry for op '{op_name}'. Add one to {CONTRACT_PATH} first."
    )


def load_dump(path: Path, shape: list[int]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("op_name")
    parser.add_argument("--dump-dir", default="cpp/test/dumps")
    args = parser.parse_args()

    entry = load_contract(args.op_name)
    dump_dir = Path(args.dump_dir)

    cpp_out = load_dump(dump_dir / f"{args.op_name}_output.bin", entry["output"])

    # np_ref is "module.path.function_name" — call it with the same inputs
    # you dumped for the C++ side to load, so both ops see identical data.
    module_path, func_name = entry["np_ref"].rsplit(".", 1)
    fn = getattr(importlib.import_module(module_path), func_name)

    inputs = {
        name: load_dump(dump_dir / f"{args.op_name}_input_{name}.bin", shape)
        for name, shape in entry["inputs"].items()
    }
    np_out = fn(**inputs)

    ok = np.allclose(cpp_out, np_out, atol=entry.get("atol", 1e-4))
    max_err = np.max(np.abs(cpp_out - np_out))
    print(f"{args.op_name}: {'PASS' if ok else 'FAIL'}  (max abs err = {max_err:.3e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
