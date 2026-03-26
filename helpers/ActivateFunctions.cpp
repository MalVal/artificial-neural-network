#include "ActivateFunctions.h"
#include <cmath>
using namespace std;

double sigmoid(double c, double x) {
    return 1.0 / (1.0 + exp(-c * x));
}

double sigmoid_derivative(double c, double x) {
    return c * sigmoid(1, x) * (1 - sigmoid(1, x));
}