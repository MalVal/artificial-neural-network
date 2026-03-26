#ifndef HIDENLAYER_H
#define HIDENLAYER_H

#include <vector>
#include "HidenNeuron.h"

class HidenLayer {
private:
    int entryNumber;
    std::vector<HidenNeuron> neurons;
    std::function<double(double, double)> activate;
    std::function<double(double, double)> derivative;
    double learningRate;
public:
    HidenLayer(int entryNumber, int neuronNumber, std::function<double(double, double)> activate, std::function<double(double, double)> derivative, double learningRate);
    std::vector<double> propagate(const vector<double>& x);
    void changeNeuronWeights(int neuronNumber, std::vector<double> newWeights);
    // GETTERS
    int getEntryNumber() const;
    int getNeuronNumber() const;
    double getLearningRate() const;
    std::function<double(double, double)> getDerivative() const;
    double getLastPotential(int neuronNumber) const;
    double getLastOutput(int neuronNumber) const;
    std::vector<double> getLastEntry(int neuronNumber) const;
    double getNeuronLastSignal(int neuronNumber) const;
    std::vector<double> getNeuronWeights(int neuronNumber) const;
    // SETTERS
    void setNeuronLastSignal(int neuronNumber, double signal);
};

#endif