#ifndef GRADIENTPERCEPTRON_H
#define GRADIENTPERCEPTRON_H

#include <vector>
#include <string>

class GradientPerceptron {
private:
    std::vector<double> weights;
    double learningRate;
    double threshold;
    int maxIter;
public:
    GradientPerceptron(int nFeatures, double lr = 0.2, double seuil = 0.125, int maxIter = 100);
    double predict(const std::vector<double>& x) const;
    void train(const std::vector<std::vector<int>>& data, const std::string& errorFile);
    void saveModel(const std::string& filename) const;
};

#endif