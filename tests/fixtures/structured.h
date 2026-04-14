/* Auto-generated from residues.yaml; do not edit. */
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    char resname[4];
    float pos[3];
    float mass;
} ResiduesStruct;

int transform(
    unsigned int X,
    unsigned int Y,
    const ResiduesStruct *residues,
    double *result
);
