#include "HiddenLayerDefinition.h"

// CONSTRUCTORS
HidenLayerDefinition::HidenLayerDefinition(int hiddenNeuronNumber, double learningRate, 
    std::function<double(double, double)> activation, 
    std::function<double(double, double)> derivative) :
    hiddenNeuronNumber(hiddenNeuronNumber),
    learningRate(learningRate),
    activation(activation),
    derivative(derivative)
{}

// GETTERS
int HidenLayerDefinition::getHiddenNeuronNumber() const {
    return this->hiddenNeuronNumber;
}

double HidenLayerDefinition::getLearningRate() const {
    return this->learningRate;
}

std::function<double(double, double)> HidenLayerDefinition::getActivation() const {
    return this->activation;
}

std::function<double(double, double)> HidenLayerDefinition::getDerivative() const {
    return this->derivative;
}