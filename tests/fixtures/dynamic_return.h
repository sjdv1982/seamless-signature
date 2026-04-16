#include <stdint.h>
#include <stdbool.h>

int transform(
    unsigned int N,
    unsigned int maxK,
    const float *signal,
    float threshold,
    unsigned int *K,
    int32_t *peak_positions,
    float *peak_values
);
