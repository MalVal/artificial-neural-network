#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <vector>
#include "HidenLayer.h"

class MultiLayerPerceptron {
private:
    std::vector<HidenLayer> hiddenLayers;
public:
    MultiLayerPerceptron(std::vector<int> layersDefinition, int entryNumber);
    void learn(std::vector<double> x, std::vector<double> d, double threshold);
};

#endif