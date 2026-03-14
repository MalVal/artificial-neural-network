#ifndef SINGLELAYERDATASETCONVERTER_H
#define SINGLELAYERDATASETCONVERTER_H

#include <vector>
#include <string>

class SingleLayerDatasetConverter {
public:
    static std::vector<std::vector<std::vector<double>>> convert(const std::vector<std::vector<double>> &dataset, int nFeatures, int nClasses);
};

#endif