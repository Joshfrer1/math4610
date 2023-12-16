#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void machineEps32();
void machineEps64();
double l_2_norm(double vector[], int size);
double l_1_norm(double vector[], int size);
double infinity_norm(double vector[], int size);
double l_2_distance(double u[], double v[], int size);
double l_1_distance(double u[], double v[], int size);
double infinity_norm_distance(double u[], double v[], int size);

int main() {
    // machineEps32();
    // machineEps64();
    double u[] = {2.0, 2.0};
    double v[] = {3.0, -1000.0};
    // l_2_norm(u, sizeof(u)/sizeof(double));
    // l_1_norm(u, sizeof(u)/sizeof(double));
    // infinity_norm(v, sizeof(v)/sizeof(double));
    // l_2_distance(u, v, sizeof(u)/sizeof(double));
    // l_1_distance(u, v, sizeof(u)/sizeof(double));
    // infinity_norm_distance(u, v, sizeof(u)/sizeof(double));
    return 0;
}

void machineEps32() {
    float previous_eps = 0.0f;
    float eps = 1.0f;
    while ((1.0f + eps) != 1.0f){
        previous_eps = eps;
        eps /= 2.0f;
    }
    printf("%e\n", previous_eps);
}

void machineEps64() {
    double previous_eps = 0.0;
    double eps = 1.0;
    while ((1.0 + eps) != 1.0){
        previous_eps = eps;
        eps /= 2.0;
    }
    printf("%e\n", previous_eps);
}

double l_2_norm(double *vector, int size) {
    double length = 0;
    for(int i = 0; i < size; i++) {
        length += pow(vector[i], 2);
    }
    printf("%lf\n", sqrt(length));
    return sqrt(length); 
}

double l_1_norm(double *vector, int size) {
    double length = 0;
    for(int i = 0; i < size; i++) {
        length += fabs(vector[i]);
    }
    printf("%lf\n", length);
    return length;
}

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

double l_2_distance(double *u, double *v, int size) {
    double distance = 0;
    for(int i = 0; i < size; i++) {
        distance += pow((u[i] - v[i]), 2);
    }
    printf("%lf\n", sqrt(distance));
    return sqrt(distance);
}

double l_1_distance(double *u, double *v, int size) {
    double distance = 0;
    for(int i = 0; i < size; i++) {
        distance += fabs(u[i] - v[i]);
    }
    printf("%lf", distance);
    return distance;
}

double infinity_norm_distance(double u[], double v[], int size) {
    double compute = infinity_norm(u, size) - infinity_norm(v, size);
    return fabs(compute);
}