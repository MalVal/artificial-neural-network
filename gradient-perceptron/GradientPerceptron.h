#ifndef GRADIENTPERCEPTRON_H
#define GRADIENTPERCEPTRON_H

#include <vector>
#include <string>

enum class StopCriterion {
    MSE_THRESHOLD,
    ZERO_ERRORS
};

class GradientPerceptron {
private:
    std::vector<double> weights;
    double learningRate;
    double threshold;
    int maxIter;
    StopCriterion criterion;
public:
    GradientPerceptron(int nFeatures, double lr, double seuil, int maxIter, StopCriterion criterion = StopCriterion::MSE_THRESHOLD);
    double predict(const std::vector<double>& x) const;
    void train(const std::vector<std::vector<double>>& data, const std::string& errorFile);
    void saveModel(const std::string& filename, bool append = false) const;
};

#endif