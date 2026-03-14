#ifndef SIMPLEPERCEPTRON_H
#define SIMPLEPERCEPTRON_H

#include <vector>
#include <string>

class SimplePerceptron {
private:
    std::vector<double> weights;
    double learningRate;
public:
    SimplePerceptron(int nFeatures, double lr = 1.0);
    int predict(const std::vector<double>& x) const;
    void train(const std::vector<std::vector<int>>& data, const std::string& errorFile);
    void saveModel(const std::string& filename, bool append = false) const;
};

#endif