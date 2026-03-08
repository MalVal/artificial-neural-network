#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "GradientPerceptron.h"

int main() {
    std::string modelCSV = "model.csv";
    std::string pointsCSV = "points.csv";
    std::string errorCSV = "errors.csv";

    // AVAILABLE DATASETS, nb features, learning rate, threshold, max iter
    std::vector<std::vector<std::string>> availableDatasets = {
        {"../../data/and.csv", "2", "0.2", "0.125001", "100"},
        {"../../data/table_2_9.csv", "2", "0.0011", "0.1251", "1000"},
        {"../../data/table_2_10.csv", "2", "0.0015", "0.1251", "1000"},
        {"../../data/table_2_11.csv", "1", "0.000167", "0.1251", "10000"}
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
    std::vector<std::vector<int>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<int> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stoi(cell));
        }
        data.push_back(row);
    }
    file.close();

    // PARAMETERS
    int nFeatures = std::stod(selectedDataset[1]);
    double learningRate = std::stod(selectedDataset[2]);
    double threshold = std::stod(selectedDataset[3]);
    int maxIter = std::stod(selectedDataset[4]);

    // LEARNING
    GradientPerceptron perceptron(nFeatures, learningRate, threshold, maxIter);
    perceptron.train(data, errorCSV);
    perceptron.saveModel(modelCSV);
    std::ofstream dataFile(pointsCSV);
    for (const auto& row : data) {
        dataFile << row[0] << ","
                 << row[1] << ","
                 << row[2] << std::endl;
    }
    dataFile.close();

    std::cout << "Training complete." << std::endl;
    std::cout << "Model saved to model.csv" << std::endl;

    // GUI
    system(("python ../../scripts/model-point.py " + pointsCSV + " " + modelCSV).c_str());
    system(("python ../../scripts/error-epoch.py " + errorCSV).c_str());

    return 0;
}