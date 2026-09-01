#include "arena.h"
#include <stdexcept>

Arena::Arena(int capacity) {
    base = new float[capacity];
    curr = base;
    checkpoint = base;
    end = base + capacity;
}

Arena::~Arena() {
    delete[] base;
}

float* Arena::alloc(int size) {
    float* res = curr;
    if (this->curr + size > this->end) {
        throw std::runtime_error("alloc: no memory available");
    }
    this->curr = this->curr + size;
    return res;
}

void Arena::reset() {
    this->curr = this->checkpoint;
}

void Arena::mark() {
    this->checkpoint = this->curr;
}