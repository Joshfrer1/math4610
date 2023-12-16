#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double l_1_norm(double *vector, int size) {
    double length = 0;
    for(int i = 0; i < size; i++) {
        length += fabs(vector[i]);
    }
    printf("%lf\n", length);
    return length;
}