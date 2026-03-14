#include "SingleLayerDatasetConverter.h"
using namespace std;

vector<vector<vector<double>>> SingleLayerDatasetConverter::convert(const std::vector<std::vector<double>> &dataset, int nFeatures, int nClasses) {
    vector<vector<vector<double>>> convertedDataset;

    for(int targetClasses = 0; targetClasses < nClasses; targetClasses++) {
        vector<vector<double>> classeDataset;
        for (const auto& row : dataset) {
            vector<double> convertedRow;
            for(int i = 0; i < nFeatures; i++) {
                convertedRow.push_back(row[i]);
            }
            convertedRow.push_back(row[nFeatures + targetClasses]);
            classeDataset.push_back(convertedRow);
        }
        convertedDataset.push_back(classeDataset);
    }

    return convertedDataset;
}