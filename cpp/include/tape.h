#pragma once
// Tape-based autograd: forward ops push a Node recording how to compute
// their gradients; backward walks the tape in reverse. See docs/roadmap.md
// Phase 5.3. Analogue of the manual reverse-order backward() calls from the
// NumPy Phase 2 implementation, just recorded instead of hardcoded.

#include <functional>
#include <vector>

struct Node {
    std::function<void()> backward;  // closure capturing whatever tensors it needs
};

extern std::vector<Node> tape;

// Records one op's backward closure onto the tape. Call this at the end of
// every forward op, after it has produced its output.
void tape_push(std::function<void()> backward);

// Walks the tape in reverse, calling every recorded backward closure.
void tape_backward();

// Clears the tape. Decide when this needs to run relative to Arena::reset()
// and the start of each training step.
void tape_reset();
