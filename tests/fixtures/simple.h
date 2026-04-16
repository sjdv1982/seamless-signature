#include <stdint.h>
#include <stdbool.h>

typedef float float32_3[3];

int transform(
    unsigned int X,
    unsigned int Y,
    const float32_3 *coords1,
    const float32_3 *coords2,
    double weight,
    double *result
);
