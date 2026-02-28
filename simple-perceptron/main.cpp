#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "SimplePerceptron.h"

int main() {
    std::string modelCSV = "model.csv", pointsCVS = "points.csv", errorCSV = "errors.csv";

    std::ifstream file("../../data/and-binary.csv");
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

    SimplePerceptron perceptron(2, 1.0);  // learning rate
    perceptron.train(data, errorCSV);
    perceptron.saveModel(modelCSV);
    std::ofstream dataFile(pointsCVS);
    for (const auto& row : data) {
        dataFile << row[0] << "," 
                 << row[1] << "," 
                 << row[2] << std::endl;
    }
    dataFile.close();

    std::cout << "Training complete." << std::endl;
    std::cout << "Model saved to model.csv" << std::endl;

    system(("python ../../scripts/model-point.py " + pointsCVS + " " + modelCSV).c_str());
    system(("python ../../scripts/error-epoch.py " + errorCSV).c_str());

    return 0;
}