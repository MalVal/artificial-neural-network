#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <vector>
#include "HidenLayer.h"

class MultiLayerPerceptron {
private:
    std::vector<HidenLayer> hiddenLayers;
public:
    MultiLayerPerceptron(std::vector<int> layersDefinition, int entryNumber);
    void learn(std::vector<std::vector<double>> X, std::vector<std::vector<double>> D, double threshold, int maxIter);
    std::vector<double> predict(std::vector<double> x);
};

#endif