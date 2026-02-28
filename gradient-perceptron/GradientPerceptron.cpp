#include "GradientPerceptron.h"
#include <iostream>
#include <fstream>
using namespace std;

GradientPerceptron::GradientPerceptron(int nFeatures, double lr, double seuil, int maxIter)
: weights(nFeatures + 1, 0.0),
      learningRate(lr),
      threshold(seuil),
      maxIter(maxIter)
{}

double GradientPerceptron::predict(const vector<double>& x) const {
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

void GradientPerceptron::train(const vector<vector<int>>& data, const string& errorFile) {
    ofstream ef(errorFile);
    if (!ef.is_open()) {
        cerr << "Cannot open error file: " << errorFile << endl;
        return;
    }

    int epoch = 0;
    while (epoch < maxIter) {
        double E = 0; // erreur quadratique totale
        vector<double> delta(weights.size(), 0.0); // correcteurs de poids

        for (const auto& row : data) {
            vector<double> x(row.begin(), row.end() - 1);
            int d = row.back();

            double y = predict(x);
            double e = d - y;

            E += 0.5 * e * e;

            // gradient
            delta[0] += learningRate * e * 1.0; // biais
            for (size_t i = 0; i < x.size(); i++) {
                delta[i+1] += learningRate * e * x[i];
            }
        }

        // Mise à jour des poids
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] += delta[i];
        }

        double Emean = E / data.size();
        ef << epoch + 1 << "," << Emean << endl;
        cout << "Epoch " << epoch+1 << " - Emean = " << Emean << endl;

        if (Emean < threshold) {
            cout << "Training finished." << endl;
            break;
        }

        epoch++;
    }

    ef.close();
}

void GradientPerceptron::saveModel(const string& filename) const {
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