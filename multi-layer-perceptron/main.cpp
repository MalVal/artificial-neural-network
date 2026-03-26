#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "../gradient-perceptron/GradientPerceptron.h"
#include "../helpers/SingleLayerDatasetConverter.h"

int main() {
    std::string modelCSV = "model.csv";
    std::string pointsCSV = "points.csv";
    std::string errorCSV = "errors.csv";

    // AVAILABLE DATASETS, nb features, learning rate, threshold, max iter, nClasses
    std::vector<std::vector<std::string>> availableDatasets = {
        {"../../data/table_3_1.csv", "2", "0.001", "0.125001", "200", "3"},
        {"../../data/table_3_5.csv", "25", "0.0011", "0.1251", "1000", "4"},
    };

    // CHOOSE DATASET
    std::cout << "Choisissez un dataset :\n";
    for (int i = 0; i < availableDatasets.size(); i++) {
        std::cout << i << ". " << availableDatasets[i][0] << "\n";
    }
    int choice;
    std::cout << "Dataset :";
    std::cin >> choice;
    if (choice < 0 || choice >= availableDatasets.size()) {
        std::cerr << "Choix invalide.\n";
        return 1;
    }
    std::vector<std::string> selectedDataset = availableDatasets[choice];

    // OPEN DATASET
    std::ifstream file(selectedDataset[0]);
    if (!file.is_open()) {
        std::cerr << "Error opening file..." << std::endl;
        return 1;
    }
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    file.close();

    // PARAMETERS
    int nFeatures = std::stod(selectedDataset[1]);
    double learningRate = std::stod(selectedDataset[2]);
    double threshold = std::stod(selectedDataset[3]);
    int maxIter = std::stod(selectedDataset[4]);
    int nClasses = std::stod(selectedDataset[5]);

    // CONVERT DATASET
    std::vector<std::vector<std::vector<double>>> convertedDataset = SingleLayerDatasetConverter::convert(data, nFeatures, nClasses);

    for(const auto& dataset : convertedDataset) {
        std::cout << "-------------------" << std::endl;
        for(const auto &row : dataset) {
            for(const auto&elem : row) {
                std::cout << elem << " "; 
            }
            std::cout << std::endl;
        }
    }

    // LEARNING
    std::ofstream dataFile(pointsCSV);
    bool firstModel = true;
    for(const auto& dataset : convertedDataset) {
        GradientPerceptron perceptron(nFeatures, learningRate, threshold, maxIter);
        perceptron.train(dataset, errorCSV);
        perceptron.saveModel(modelCSV, !firstModel);
        firstModel = false;
        for (const auto& row : data) {
            dataFile << row[0] << ","
                    << row[1] << ","
                    << row[2] << std::endl;
        }
    }
    dataFile.close();

    std::cout << "Training complete." << std::endl;
    std::cout << "Model saved to model.csv" << std::endl;

    // GUI
    system(("python ../../scripts/multi-model-point.py " + pointsCSV + " " + modelCSV).c_str());
    system(("python ../../scripts/error-epoch.py " + errorCSV).c_str());

    return 0;
}