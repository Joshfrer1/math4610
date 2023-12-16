#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double l_2_distance(double *u, double *v, int size) {
    double distance = 0;
    for(int i = 0; i < size; i++) {
        distance += pow((u[i] - v[i]), 2);
    }
    printf("%lf\n", sqrt(distance));
    return sqrt(distance);
}