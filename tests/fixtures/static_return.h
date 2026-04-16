#include <stdint.h>
#include <stdbool.h>

typedef float float32_3[3];

int transform(
    unsigned int N,
    const float32_3 *coords,
    float *dist
);
