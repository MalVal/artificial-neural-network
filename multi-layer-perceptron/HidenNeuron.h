#ifndef HIDENNEURON_H
#define HIDENNEURON_H

#include <vector>
#include <string>
#include <functional>
#include <cmath>

class HidenNeuron {
private:
    std::vector<double> weights;
    std::function<double(double, double)> activation;
    double lastPotential;
    std::vector<double> lastEntry;
    double lastOutput;
public:
    double lastErrorSignal;
    HidenNeuron(std::vector<double> &w, std::function<double(double, double)> activation);
    double predict(const std::vector<double>& x) const;
    double getOutput(const std::vector<double>& data);
    void changeWeights(std::vector<double> &newWeights);
    // GETTERS
    double getLastPotential() const;
    double getLastOutput() const;
    std::vector<double> getWeights() const;
    std::vector<double> getLastEntry() const;
};

#endif