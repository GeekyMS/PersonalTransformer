// Sanity check for Tensor::slice — pure indexing math, no NumPy reference
// needed (unlike the other test/op_*.cpp binaries, see test/README.md).
#include <cassert>
#include <cstdio>
#include <vector>

#include "tensor.h"

int main() {
    // Shape (2,3,4,5): data[i] = i, so at({b,h,i,j}) == flat index == value.
    const int B = 2, H = 3, T = 4, D = 5;
    std::vector<float> data(B * H * T * D);
    std::vector<float> grad(B * H * T * D, 0.0f);
    for (int i = 0; i < (int)data.size(); i++) data[i] = (float)i;

    Tensor x(data.data(), {B, H, T, D});
    x.grad = grad.data();

    // Fix (b=1, h=2): should yield a (T,D) view starting at flat index
    // (1*H + 2)*T*D = 5*4*5 = 100.
    Tensor xs = x.slice({1, 2});

    assert(xs.shape == (std::vector<int>{T, D}));
    assert(xs.strides == (std::vector<int>{D, 1}));
    assert(xs.offset == 100);

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < D; j++) {
            assert(xs.at({i, j}) == x.at({1, 2, i, j}));
        }
    }

    // grad must alias the parent's buffer: writing through the slice should
    // be visible through the parent at the same logical index.
    xs.grad_at({0, 0}) = 42.0f;
    assert(x.grad_at({1, 2, 0, 0}) == 42.0f);

    // Fixing a single leading axis should drop just that one: (H,T,D).
    Tensor xb = x.slice({1});
    assert(xb.shape == (std::vector<int>{H, T, D}));
    assert(xb.strides == (std::vector<int>{T * D, D, 1}));
    assert(xb.offset == H * T * D);

    std::printf("op_slice: all checks passed\n");
    return 0;
}
