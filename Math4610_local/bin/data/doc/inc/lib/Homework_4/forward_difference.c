#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static double fx_x_cubed(double x) {
    return x * x * x;
}

double forward_difference(double x, double h) {
    return (fx_x_cubed(x + h) - fx_x_cubed(x))/h;
}