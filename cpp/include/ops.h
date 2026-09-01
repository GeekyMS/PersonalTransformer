#pragma once
// Forward ops. Each computes its forward result into `out` and pushes its
// own backward closure onto the tape (see tape.h). See docs/roadmap.md
// Phase 5.4 for the port protocol: implement, dump, diff against NumPy,
// only then move to the next op.

#include "tensor.h"

// out = A @ B. A: (M,K), B: (K,N), out: (M,N). Caller allocates out.
void matmul(const Tensor& A, const Tensor& B, Tensor& out);
