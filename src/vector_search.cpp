#include "vector_search.h"
#include <cmath>
#include <algorithm>
#include <cstdint>

float dot_product(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

int32_t dot_product_int8(
    const int8_t* a,
    const int8_t* b,
    size_t dim
) {
    int32_t sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<int32_t>(a[i]) *
               static_cast<int32_t>(b[i]);
    }
    return sum;
}

void quantize_vector(
    const float* input,
    int8_t* output,
    float& scale,
    size_t dim
) {
    float max_abs = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        max_abs = std::max(max_abs, std::abs(input[i]));
    }

    scale = max_abs / 127.0f;
    if (scale == 0.0f) scale = 1e-8f;

    for (size_t i = 0; i < dim; ++i) {
        output[i] = static_cast<int8_t>(
            std::round(input[i] / scale)
        );
    }
}