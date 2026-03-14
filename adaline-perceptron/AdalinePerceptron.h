#ifndef ADALINEPERCEPTRON_H
#define ADALINEPERCEPTRON_H

#include <vector>
#include <string>

enum class StopCriterion {
    MSE_THRESHOLD,   // Arrêt si erreur quadratique moyenne < seuil
    ZERO_ERRORS      // Arrêt si nombre d'erreurs de classification = 0
};

class AdalinePerceptron {
private:
    std::vector<double> weights;
    double learningRate;
    double threshold;
    int maxIter;
    StopCriterion criterion;
public:
    AdalinePerceptron(int nFeatures, double lr, double seuil, int maxIter, StopCriterion criterion = StopCriterion::MSE_THRESHOLD);
    double predict(const std::vector<double>& x) const;
    void train(const std::vector<std::vector<double>>& data, const std::string& errorFile);
    void saveModel(const std::string& filename, bool append = false) const;
};

#endif