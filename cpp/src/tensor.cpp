#include "tensor.h"
#include <stdexcept>

Tensor::Tensor(float* data, std::vector<int> shape)
    : data(data), shape(std::move(shape)), strides(this->shape.size()), offset(0), grad(nullptr) {
    int len = this->shape.size();
    this->strides[len - 1] = 1;
    for (int i = len - 2; i >= 0; i --){
        this->strides[i] = this->strides[i + 1] * this->shape[i + 1];
    }
}

Tensor::Tensor(float* data, std::vector<int> shape, std::vector<int> strides, int offset)
    : data(data), shape(std::move(shape)), strides(std::move(strides)), offset(offset), grad(nullptr) {}

float& Tensor::at(const std::vector<int>& idx) const {
    int temp = 0;
    for(int i = 0; i < (int)idx.size(); i++){
        int k = idx[i];
        temp += k * this->strides[i];
    }
    return this->data[temp + this->offset];
}

float& Tensor::grad_at(const std::vector<int>& idx) const {
    int temp = 0;
    for(int i = 0; i < (int)idx.size(); i++){
        int k = idx[i];
        temp += k * this->strides[i];
    }
    return this->grad[temp + this->offset];
}

Tensor Tensor::transpose(int i, int j) const {
    std::vector<int> newShape(this->shape);
    newShape[i] = this->shape[j];
    newShape[j] = this->shape[i];

    std::vector<int> newStrides(this->strides);
    newStrides[i] = this->strides[j];
    newStrides[j] = this->strides[i];

    Tensor res = Tensor(this->data, newShape, newStrides, this->offset);
    return res;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int ShapeProduct = 1;
    for (int x : shape) ShapeProduct *= x;

    int NewProduct = 1;
    for (int x : new_shape) NewProduct *= x;



    if(ShapeProduct != NewProduct){
        throw std::runtime_error("reshape: incompatible element count");
    }

    int temp = 1;
    for(int i = this->shape.size() - 1; i >= 0; i--){
        if(temp != this->strides[i]){
            throw std::runtime_error("reshape: tensor is not contiguous, cannot reshape without a copy");
        }
        temp *= this->shape[i];
    }

    std::vector<int> newStrides(new_shape.size());
    newStrides[(int)new_shape.size() - 1] = 1;

    for(int i = (int)new_shape.size() - 2; i >= 0; i--){
        newStrides[i] = newStrides[i + 1] * new_shape[i + 1];
    }

    Tensor res = Tensor(this->data, new_shape, newStrides, this->offset);

    return res;
}

Tensor make_tensor(Arena& arena, const std::vector<int>& shape, bool needs_grad) {
    int total = 1;
    for (int s : shape) total *= s;

    Tensor t(arena.alloc(total), shape);
    if (needs_grad) {
        t.grad = arena.alloc(total);
        for (int i = 0; i < total; i++) t.grad[i] = 0.0f;
    }
    return t;
}

Tensor slice(std::vector<int> fixed_indices){
    int old_offset = this->offset;
    int new_offset = old_offset;

    int temp = 0;

    for(int k = 0; k < fixed_indices.size(); k++){
        temp += this->strides[k] * fixed_indices[i];
    }
    new_offset += temp



    Tensor res = Tensor(t.data, )
}
