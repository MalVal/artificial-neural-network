#include "ActivateFunctions.h"
#include <cmath>
using namespace std;

double sigmoid(double c, double x) {
    return 1.0 / (1.0 + exp(-c * x));
}

double sigmoid_derivative(double c, double x) {
    double s = sigmoid(c, x);
    return c * s * (1.0 - s);
}

double identity(double c, double x) {
    return x;
}

double identity_derivative(double c, double x) {
    return 1.0;
}

double tanh_act(double c, double x) {
    return c * tanh(x);
}

double tanh_derivative(double c, double x) {
    double t = tanh(x);
    return c * (1.0 - t * t);
}