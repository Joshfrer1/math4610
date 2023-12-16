#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static double infinity_norm(double *vector, int size) {
    double max = vector[0];
    for(int i = 0; i < size; i++) {
        double abs_i = fabs(vector[i]);
        if (max < abs_i) {
            max = abs_i;
        }
    }
    return max;
}

double infinity_norm_distance(double u[], double v[], int size) {
    double compute = infinity_norm(u, size) - infinity_norm(v, size);
    return fabs(compute);
}

