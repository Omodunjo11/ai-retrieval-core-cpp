#include "vector_search.h"

float dot_product(const float* a, const float* b, size_t dim) {
    float result = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
