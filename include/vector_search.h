#pragma once
#include <cstddef>
#include <cstdint>

float dot_product(const float* a, const float* b, size_t dim);

int32_t dot_product_int8(
    const int8_t* a,
    const int8_t* b,
    size_t dim
);

void quantize_vector(
    const float* input,
    int8_t* output,
    float& scale,
    size_t dim
);