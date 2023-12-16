#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double l_1_distance(double *u, double *v, int size) {
    double distance = 0;
    for(int i = 0; i < size; i++) {
        distance += fabs(u[i] - v[i]);
    }
    printf("%lf", distance);
    return distance;
}