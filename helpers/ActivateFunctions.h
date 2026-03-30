#ifndef ACTIVATEFUNCTIONS_H
#define ACTIVATEFUNCTIONS_H

// Sigmoid
double sigmoid(double c, double x);
double sigmoid_derivative(double c, double x);

// Identity (linear)
double identity(double c, double x);
double identity_derivative(double c, double x);

// Hyperbolic tangent
double tanh_act(double c, double x);
double tanh_derivative(double c, double x);

#endif