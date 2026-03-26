#include "HidenNeuron.h"
#include <iostream>
using namespace std;

HidenNeuron::HidenNeuron(vector<double> &w, function<double(double, double)> activate)
: weights(w),
    activation(activate)
{}

double HidenNeuron::predict(const vector<double>& x) const {
    if (x.size() != weights.size() - 1) {
        cerr << "Error: input size does not match number of features" << endl;
        return -1;
    }
    double sum = weights[0];
    for (size_t i = 0; i < x.size(); i++) {
        sum += weights[i+1] * x[i];
    }
    return sum;
}

double HidenNeuron::getOutput(const vector<double>& x) {
    lastEntry = x;
    lastPotential = predict(lastEntry);
    lastOutput = activation(1.0, lastPotential);
    return lastOutput;
}

double HidenNeuron::getLastPotential() const {
    return lastPotential;
}

vector<double> HidenNeuron::getLastEntry() const {
    return lastEntry;
}

double HidenNeuron::getLastOutput() const {
    return lastOutput;
}

vector<double> HidenNeuron::getWeights() const {
    return weights;
}

void HidenNeuron::changeWeights(vector<double> &newWeights) {
    for(int i = 0; i < weights.size(); i++) {
        weights[i] += newWeights[i];
    }
}