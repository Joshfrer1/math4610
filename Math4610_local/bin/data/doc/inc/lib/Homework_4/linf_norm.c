#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double infinity_norm(double *vector, int size) {
    double max = vector[0];
    for(int i = 0; i < size; i++) {
        double abs_i = fabs(vector[i]);
        if (max < abs_i) {
            max = abs_i;
        }
    }
    return max;
}