#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "MultiLayerPerceptron.h"
#include "HiddenLayerDefinition.h"
#include "../helpers/ActivateFunctions.h"

using namespace std;

int main() {

    // AVAILABLE DATASETS
    // Data, Entry number, Learning rate, threshold, Itermax, Output number
    std::vector<std::vector<std::string>> availableDatasets = {
        {"../../data/xor.csv", "2", "0.8", "0.001", "2000", "1"},
        {"../../data/table_4_12.csv", "2", "0.5", "0.001", "2000", "1"},
        {"../../data/table_4_14.csv", "2", "1.2", "0.001", "2000", "3"},
        {"../../data/table_4_17.csv", "1", "0.001", "0.05", "20000", "1"}
    };

    // CHOOSE DATASET
    cout << "Choisissez un dataset :\n";
    for (int i = 0; i < availableDatasets.size(); i++) {
        cout << i << ". " << availableDatasets[i][0] << "\n";
    }

    int choice;
    cout << "Dataset :";
    cin >> choice;

    if (choice < 0 || choice >= availableDatasets.size()) {
        cerr << "Choix invalide.\n";
        return 1;
    }

    vector<string> selectedDataset = availableDatasets[choice];

    int nbInputs = stoi(selectedDataset[1]);
    int nbOutputs = stoi(selectedDataset[5]);

    // OPEN DATASET
    ifstream file(selectedDataset[0]);
    if (!file.is_open()) {
        cerr << "Error opening file..." << endl;
        return 1;
    }

    string line;
    vector<vector<double>> X;
    vector<vector<double>> D;

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }

        if (row.size() != nbInputs + nbOutputs) continue;

        vector<double> x(row.begin(), row.begin() + nbInputs);
        vector<double> d(row.begin() + nbInputs, row.end());

        X.push_back(x);
        D.push_back(d);
    }

    file.close();

    // ARCHITECTURE
    MultiLayerPerceptron *mlp = NULL;
    /* REGRESSION*/
    if(choice == 3) {
        mlp = new MultiLayerPerceptron
        (
            {
                HidenLayerDefinition(nbInputs, stod(selectedDataset[2]), sigmoid, sigmoid_derivative), 
                HidenLayerDefinition(10, stod(selectedDataset[2]), sigmoid, sigmoid_derivative), 
                HidenLayerDefinition(nbOutputs, stod(selectedDataset[2]), identity, identity_derivative)
            }
        );
    }
    else {
        mlp = new MultiLayerPerceptron
        (
            {
                HidenLayerDefinition(nbInputs, stod(selectedDataset[2]), sigmoid, sigmoid_derivative), 
                HidenLayerDefinition(5, stod(selectedDataset[2]), sigmoid, sigmoid_derivative), 
                HidenLayerDefinition(nbOutputs, stod(selectedDataset[2]), sigmoid, sigmoid_derivative)
            }
        );
    }

    // TRAINING
    mlp->learn(X, D, stod(selectedDataset[3]), stoi(selectedDataset[4]));

    // DISPLAY RESULTS
    cout << "=== Results ===" << endl;

    for (int k = 0; k < (int)X.size(); k++) {
        vector<double> output = mlp->predict(X[k]);

        // inputs
        for (int i = 0; i < nbInputs; i++) {
            cout << X[k][i];
            if (i < nbInputs - 1) cout << " ";
        }

        cout << " --> ";

        // output interpretation
        if (nbOutputs == 1) {
            cout << output[0];
            cout << " (attendu: " << D[k][0] << ")";
        } else {
            // multi-class → afficher argmax
            int predictedClass = 0;
            for (int i = 1; i < nbOutputs; i++) {
                if (output[i] > output[predictedClass])
                    predictedClass = i;
            }

            cout << "classe = " << predictedClass << " (";

            for (int i = 0; i < nbOutputs; i++) {
                cout << output[i];
                if (i < nbOutputs - 1) cout << ",";
            }

            cout << ")";
        }

        cout << endl;
    }

    // EXPORT FOR PYTHON (only if 2D input for visualization)
    if (nbInputs == 1 || nbInputs == 2) {

        if (X.empty()) {
            cerr << "Dataset vide !" << endl;
            return 1;
        }

        ofstream out("grid_predictions.csv");

        double x_min = X[0][0], x_max = X[0][0];
        double y_min = 0, y_max = 0;

        // calcul bornes
        for (auto &p : X) {
            x_min = min(x_min, p[0]);
            x_max = max(x_max, p[0]);

            if (nbInputs == 2) {
                y_min = (p == X[0]) ? p[1] : min(y_min, p[1]);
                y_max = (p == X[0]) ? p[1] : max(y_max, p[1]);
            }
        }

        // marge
        double margin = 0.1;
        x_min -= margin;
        x_max += margin;

        if (nbInputs == 2) {
            y_min -= margin;
            y_max += margin;
        }

        double step = 0.01;

        // génération grid
        if (nbInputs == 2) {
            for (double x = x_min; x <= x_max; x += step) {
                for (double y = y_min; y <= y_max; y += step) {
                    vector<double> input = {x, y};
                    vector<double> output = mlp->predict(input);

                    out << x << "," << y;
                    for (double o : output) out << "," << o;
                    out << "\n";
                }
            }
        } else { // 1D
            for (double x = x_min; x <= x_max; x += step) {
                vector<double> input = {x};
                vector<double> output = mlp->predict(input);

                out << x << "," << output[0] << "\n";
            }
        }

        out.close();

        // training data
        ofstream train_out("training_data.csv");
        for (int i = 0; i < X.size(); i++) {
            for (int j = 0; j < nbInputs; j++) {
                train_out << X[i][j] << ",";
            }
            for (int j = 0; j < nbOutputs; j++) {
                train_out << D[i][j];
                if (j < nbOutputs - 1) train_out << ",";
            }
            train_out << "\n";
        }
        train_out.close();

        system("python ../../scripts/multi-layer.py grid_predictions.csv training_data.csv");
    }

    // END
    return 0;
}