#include "HidenLayer.h"
#include <iostream>
#include <random>
using namespace std;

HidenLayer::HidenLayer(int entryNumber, int neuronNumber, function<double(double, double)> activate, function<double(double, double)> derivative, double learningRate)
: entryNumber(entryNumber),
    activate(activate),
    derivative(derivative),
    learningRate(learningRate)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);
    for(int i = 0; i < neuronNumber; i++) {
        // Weights initialisation
        vector<double> weights;
        for(int j = 0; j < entryNumber + 1; j++) {
            weights.push_back(dist(gen));
        }
        // Create the neuron
        neurons.push_back(HidenNeuron(weights, activate));
    }
}

vector<double> HidenLayer::propagate(const vector<double>& x) {
    // Retreive the output of each neuron of this layer
    vector<double> neuronOutputs;
    neuronOutputs.reserve(neurons.size());
    for(auto &neuron : neurons) {
        neuronOutputs.push_back(neuron.getOutput(x));
    }
    return neuronOutputs;
}

double HidenLayer::getLastPotential(int neuronNumber) const {
    return neurons[neuronNumber].getLastPotential();
}

std::vector<double> HidenLayer::getLastEntry(int neuronNumber) const {
    return neurons[neuronNumber].getLastEntry();
}

double HidenLayer::getLastOutput(int neuronNumber) const {
    return neurons[neuronNumber].getLastOutput();
}

std::function<double(double, double)> HidenLayer::getDerivative() const {
    return derivative;
}

void HidenLayer::changeNeuronWeights(int neuronNumber, std::vector<double> newWeights) {
    neurons[neuronNumber].changeWeights(newWeights);
}

int HidenLayer::getEntryNumber() const {
    return entryNumber;
}

int HidenLayer::getNeuronNumber() const {
    return neurons.size();
}

double HidenLayer::getLearningRate() const {
    return learningRate;
}

double HidenLayer::getNeuronLastSignal(int neuronNumber) const {
    return neurons[neuronNumber].lastErrorSignal;
}

vector<double> HidenLayer::getNeuronWeights(int neuronNumber) const {
    return neurons[neuronNumber].getWeights();
}

// SETTERS

void HidenLayer::setNeuronLastSignal(int neuronNumber, double signal) {
    neurons[neuronNumber].lastErrorSignal = signal;
}