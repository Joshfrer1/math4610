#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double l_2_norm(double *vector, int size) {
    double length = 0;
    for(int i = 0; i < size; i++) {
        length += pow(vector[i], 2);
    }
    printf("%lf\n", sqrt(length));
    return sqrt(length); 
}