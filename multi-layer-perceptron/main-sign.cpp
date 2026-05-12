#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include "MultiLayerPerceptron.h"
#include "HiddenLayerDefinition.h"
#include "../helpers/ActivateFunctions.h"

using namespace std;

int main() {

    // OPEN DATASET
    ifstream file("../../data/LangageDesSignes/data_formatted.csv");
    if (!file.is_open()) {
        cerr << "Erreur : impossible d'ouvrir le fichier." << endl;
        return 1;
    }

    int nbInputs  = 42;
    int nbOutputs = 5;

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

        if ((int)row.size() != nbInputs + nbOutputs) continue;

        vector<double> x(row.begin(), row.begin() + nbInputs);
        vector<double> d(row.begin() + nbInputs, row.end());

        X.push_back(x);
        D.push_back(d);
    }
    file.close();

    cout << "Donnees chargees : " << X.size() << " exemples." << endl;

    // SPLIT PAR CLASSE : 50 learning + 10 validation par classe
    // Les données sont groupées par classe (60 par classe, 5 classes)
    vector<vector<double>> X_learn, D_learn;
    vector<vector<double>> X_val,   D_val;

    int nbClasses       = 5;
    int totalPerClass   = 60;
    int learnPerClass   = 50;
    int valPerClass     = 10;

    for (int c = 0; c < nbClasses; c++) {
        int start = c * totalPerClass;

        for (int i = 0; i < learnPerClass; i++) {
            X_learn.push_back(X[start + i]);
            D_learn.push_back(D[start + i]);
        }
        for (int i = learnPerClass; i < totalPerClass; i++) {
            X_val.push_back(X[start + i]);
            D_val.push_back(D[start + i]);
        }
    }

    // Mélange du learning dataset
    vector<int> indices(X_learn.size());
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(42);
    shuffle(indices.begin(), indices.end(), rng);

    vector<vector<double>> X_learn_shuffled, D_learn_shuffled;
    for (int idx : indices) {
        X_learn_shuffled.push_back(X_learn[idx]);
        D_learn_shuffled.push_back(D_learn[idx]);
    }

    cout << "Learning set : " << X_learn_shuffled.size() << " exemples." << endl;
    cout << "Validation set : " << X_val.size() << " exemples." << endl;

    // ARCHITECTURE DU RÉSEAU
    // Entrée : 42 neurones (21 points x,y)
    // Couche cachée 1 : 20 neurones (sigmoid)
    // Couche cachée 2 : 10 neurones (sigmoid)
    // Sortie : 5 neurones (sigmoid, one-hot par classe)
    double lr = 0.1;

    MultiLayerPerceptron* mlp = new MultiLayerPerceptron(
        42,
        {
            //HidenLayerDefinition(42, lr, sigmoid, sigmoid_derivative),
            //HidenLayerDefinition(20, lr, sigmoid, sigmoid_derivative),
            HidenLayerDefinition(10, lr, sigmoid, sigmoid_derivative),
            HidenLayerDefinition(5,  lr, sigmoid, sigmoid_derivative)
        }
    );

    // APPRENTISSAGE
    cout << "\n=== Apprentissage ===" << endl;
    mlp->learn(X_learn_shuffled, D_learn_shuffled, 0.01, 10000);

    // ÉVALUATION SUR LE LEARNING SET
    cout << "\n=== Evaluation sur le learning set ===" << endl;
    int correctLearn = 0;
    for (int k = 0; k < (int)X_learn_shuffled.size(); k++) {
        vector<double> output = mlp->predict(X_learn_shuffled[k]);

        int predicted = 0;
        for (int i = 1; i < nbOutputs; i++)
            if (output[i] > output[predicted]) predicted = i;

        int expected = 0;
        for (int i = 1; i < nbOutputs; i++)
            if (D_learn_shuffled[k][i] > D_learn_shuffled[k][expected]) expected = i;

        if (predicted == expected) correctLearn++;
    }
    double learnAcc = 100.0 * correctLearn / X_learn_shuffled.size();
    cout << "Precision learning : " << correctLearn << "/" << X_learn_shuffled.size()
         << " (" << learnAcc << "%)" << endl;

    // ÉVALUATION SUR LE VALIDATION SET
    cout << "\n=== Evaluation sur le validation set ===" << endl;

    // Matrice de confusion 5x5
    vector<vector<int>> confusion(nbClasses, vector<int>(nbClasses, 0));
    string classNames[] = {"A", "B", "C", "D", "E"};

    int correctVal = 0;
    for (int k = 0; k < (int)X_val.size(); k++) {
        vector<double> output = mlp->predict(X_val[k]);

        int predicted = 0;
        for (int i = 1; i < nbOutputs; i++)
            if (output[i] > output[predicted]) predicted = i;

        int expected = 0;
        for (int i = 1; i < nbOutputs; i++)
            if (D_val[k][i] > D_val[k][expected]) expected = i;

        confusion[expected][predicted]++;
        if (predicted == expected) correctVal++;
    }

    double valAcc = 100.0 * correctVal / X_val.size();
    cout << "Precision validation : " << correctVal << "/" << X_val.size()
         << " (" << valAcc << "%)" << endl;

    // Précision par classe
    cout << "\nPrecision par classe :" << endl;
    for (int i = 0; i < nbClasses; i++) {
        int total = valPerClass;
        int correct = confusion[i][i];
        cout << "  " << classNames[i] << " : " << correct << "/" << total
             << " (" << (100.0 * correct / total) << "%)" << endl;
    }

    delete mlp;
    return 0;
}