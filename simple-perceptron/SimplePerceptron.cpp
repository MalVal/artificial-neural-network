#include "SimplePerceptron.h"
#include <iostream>
#include <fstream>
using namespace std;

SimplePerceptron::SimplePerceptron(int nFeatures, double lr) 
: weights(nFeatures + 1, 0.0), learningRate(lr)
{
    // Tous les poids initialisés à 0
}

int SimplePerceptron::predict(const vector<double>& x) const {
    if (x.size() != weights.size() - 1) {
        cerr << "Error: input size does not match number of features" << endl;
        return -1;
    }
    double sum = weights[0]; // biais
    for (size_t i = 0; i < x.size(); i++) {
        sum += weights[i+1] * x[i];
    }
    return (sum >= 0) ? 1 : 0;
}

void SimplePerceptron::train(const vector<vector<int>>& data, const string& errorFilename) {
    ofstream errorFile(errorFilename);
    if (!errorFile.is_open()) {
        cerr << "Cannot open error file: " << errorFilename << endl;
        return;
    }

    int epoch = 1;

    while (true) {
        int nbError = 0;

        for (const auto& row : data) {

            // Extraire les features
            vector<double> x(row.begin(), row.end() - 1);
            int d = row.back(); // sortie désirée

            int y = predict(x);
            int e = d - y;

            if (e != 0) {
                nbError++;

                // Mettre à jour le biais
                weights[0] += learningRate * e;

                // Mettre à jour les autres poids
                for (size_t i = 0; i < x.size(); i++) {
                    weights[i+1] += learningRate * e * x[i];
                }
            }
        }

        // Écriture des erreurs
        errorFile << epoch << "," << nbError << endl;

        cout << "Epoch " << epoch << " - Errors: " << nbError << endl;

        if (nbError == 0) {
            cout << "Training finished." << endl;
            break;
        }

        epoch++;
    }

    errorFile.close();
}

void SimplePerceptron::saveModel(const string& filename) const {
    ofstream modelFile(filename);
    if (!modelFile.is_open()) {
        cerr << "Cannot open model file: " << filename << endl;
        return;
    }
    for (size_t i = 0; i < weights.size(); i++) {
        modelFile << weights[i];
        if (i != weights.size() - 1)
            modelFile << ",";
    }
    modelFile << endl;
    modelFile.close();
}