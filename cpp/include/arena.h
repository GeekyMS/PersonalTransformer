#pragma once
// Bump allocator over one big preallocated buffer. See docs/roadmap.md Phase 5.2.
// One Arena is the single source of truth for memory during training — it is
// never copied, only passed around by reference/pointer.

struct Arena {
    float* base;
    float* curr;
    float* checkpoint;
    float* end;

    explicit Arena(int capacity);
    ~Arena();

    // Non-copyable: two Arenas must never share ownership of the same buffer.
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Hands out a chunk of `size` floats, advancing the bump pointer.
    // Throws if the arena doesn't have room left.
    float* alloc(int size);

    // Rewinds the bump pointer to the checkpoint, freeing everything allocated
    // since the last reset (or since construction). Call at the start of
    // each training step.
    void reset();

    void mark();
};
