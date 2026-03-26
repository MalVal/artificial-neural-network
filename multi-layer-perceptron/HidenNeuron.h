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
public:
    HidenNeuron(std::vector<double> &w, std::function<double(double, double)> activation);
    double predict(const std::vector<double>& x) const;
    double getOutput(const std::vector<double>& data);
    double getPotential() const;
    std::vector<double> getLastEntry() const;
    void changeWeights(std::vector<double> &newWeights);
};

#endif