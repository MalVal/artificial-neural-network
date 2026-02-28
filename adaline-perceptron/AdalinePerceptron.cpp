#include "AdalinePerceptron.h"
#include <iostream>
#include <fstream>
using namespace std;

AdalinePerceptron::AdalinePerceptron(int nFeatures, double lr, double seuil, int maxIter, StopCriterion criterion)
: weights(nFeatures + 1, 0.0),
    learningRate(lr),
    threshold(seuil),
    maxIter(maxIter),
    criterion(criterion)
{}

double AdalinePerceptron::predict(const vector<double>& x) const {
    if (x.size() != weights.size() - 1) {
        cerr << "Error: input size does not match number of features" << endl;
        return -1;
    }
    double sum = weights[0] * 1.0;  // biais
    for (size_t i = 0; i < x.size(); i++) {
        sum += weights[i+1] * x[i];
    }
    return sum; // sortie linéaire
}

void AdalinePerceptron::train(const vector<vector<int>>& data, const string& errorFile) {
    ofstream ef(errorFile);
    if (!ef.is_open()) {
        cerr << "Cannot open error file: " << errorFile << endl;
        return;
    }

    int epoch = 0;
    while (epoch < maxIter) {
        for (const auto& row : data) {
            vector<double> x(row.begin(), row.end() - 1);
            int d = row.back();
            double y = predict(x);
            double e = d - y;
            weights[0] += learningRate * e;
            for (size_t i = 0; i < x.size(); i++)
                weights[i+1] += learningRate * e * x[i];
        }

        double E = 0;
        int nbErrors = 0;
        for (const auto& row : data) {
            vector<double> x(row.begin(), row.end() - 1);
            int d = row.back();
            double y = predict(x);
            double e = d - y;
            E += 0.5 * e * e;
            int yClass = (y >= 0.0) ? 1 : -1;
            if (yClass != d) nbErrors++;
        }

        double Emean = E / data.size();
        ef << epoch + 1 << "," << Emean << "," << nbErrors << endl;
        cout << "Epoch " << epoch+1 << " - Emean = " << Emean
             << " - NbErrors = " << nbErrors << endl;

        // Critère d'arrêt
        bool stop = false;
        if (criterion == StopCriterion::MSE_THRESHOLD && Emean < threshold)
            stop = true;
        else if (criterion == StopCriterion::ZERO_ERRORS && nbErrors == 0)
            stop = true;

        if (stop) {
            cout << "Training finished." << endl;
            break;
        }

        epoch++;
    }

    ef.close();
}

void AdalinePerceptron::saveModel(const string& filename) const {
    ofstream modelFile(filename);
    if (!modelFile.is_open()) {
        cerr << "Cannot open model file: " << filename << endl;
        return;
    }
    for (size_t i = 0; i < weights.size(); i++) {
        modelFile << weights[i];
        if (i != weights.size() - 1) modelFile << ",";
    }
    modelFile << endl;
    modelFile.close();
}