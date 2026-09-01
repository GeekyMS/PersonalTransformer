#pragma once
// The Tensor view type: a shape/strides/offset descriptor over a flat float
// buffer it does not own. See docs/roadmap.md Phase 5.1.

#include <vector>

struct Tensor {
    float*           data;     // not owned
    std::vector<int> shape;
    std::vector<int> strides;
    int              offset;
    float*           grad;

    // Builds a fresh, contiguous (row-major) view over `data` with the given
    // shape: computes strides, sets offset = 0.
    Tensor(float* data, std::vector<int> shape);

    // Builds a view directly from already-known shape/strides/offset — no
    // computation, just stores what it's given. Used by transpose/reshape,
    // which construct valid (possibly non-contiguous) shape/strides by hand.
    Tensor(float* data, std::vector<int> shape, std::vector<int> strides, int offset);

    // Element access via shape-space indices, e.g. t.at({b, h, i, j}).
    float& at(const std::vector<int>& idx) const;

    float& grad_at(const std::vector<int>& idx) const;

    // Returns a new Tensor with axes i and j swapped. Same `data`, no copy.
    Tensor transpose(int i, int j) const;

    // Returns a new Tensor with the given shape. Only legal without copying
    // if the current view is contiguous in the order the new shape implies —
    // decide what that condition is and assert it for now (copying reshape
    // can come later if you need it).
    Tensor reshape(const std::vector<int>& new_shape) const;
};
