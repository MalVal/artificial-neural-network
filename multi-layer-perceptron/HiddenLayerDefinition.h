#ifndef HIDENLAYERDEFINITION_H
#define HIDENLAYERDEFINITION_H

#include <vector>
#include <functional>

class HidenLayerDefinition {
private:
    int hiddenNeuronNumber;
    double learningRate;
    std::function<double(double, double)> activation;
    std::function<double(double, double)> derivative;
public:
    // CONSTRUCTORS
    HidenLayerDefinition(int hiddenNeuronNumber, double learningRate, 
        std::function<double(double, double)> activation, 
        std::function<double(double, double)> derivative);
    // GETTERS
    int getHiddenNeuronNumber() const;
    double getLearningRate() const;
    std::function<double(double, double)> getActivation() const;
    std::function<double(double, double)> getDerivative() const;
};

#endif