# Port protocol (roadmap 5.4)

Op by op, diffed against the NumPy reference at each step:

1. Write the C++ op in `cpp/src/` (declared in the matching header under `cpp/src/`).
2. Write a tiny test binary `cpp/test/op_<name>.cpp` that:
   - loads its inputs via `load_binary()` (from `cpp/include/dump.h`) — the same
     input files the NumPy side will use, so both sides see identical data
   - runs the op
   - dumps the output via `dump_binary()` to `cpp/test/dumps/<name>_output.bin`
3. Add an entry to `cpp/test/contract.json` naming the op, its NumPy reference
   function, and the shapes of every input/output.
4. Generate the input dumps from Python (random or fixed-seed data, written with
   `.tofile()`), run the C++ test binary, then:
   ```
   python cpp/test/compare.py <op_name>
   ```
5. Only move to the next op once this passes.

CMake picks up every `cpp/test/op_*.cpp` automatically and builds it as its own
binary — no CMakeLists.txt edits needed as you add ops.
